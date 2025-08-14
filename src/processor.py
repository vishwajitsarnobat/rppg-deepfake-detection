# src/processor.py

import cv2
import dlib
import numpy as np
import logging
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import os
from torchvision import transforms
from typing import List, Dict, Any, Optional

from . import config

# --- dlib Model Loading ---
try:
    if not os.path.exists(config.DLIB_MODEL_PATH):
        raise FileNotFoundError(f"Dlib model not found at {config.DLIB_MODEL_PATH}. Please ensure it's in the 'models' directory.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.DLIB_MODEL_PATH)
except (FileNotFoundError, RuntimeError) as e:
    logging.error(f"FATAL: Failed to initialize Dlib. {e}")
    detector = None
    predictor = None


def _extract_face_rois(frame: np.ndarray) -> Optional[List[np.ndarray]]:
    """Detects the largest face and extracts 22 specific Regions of Interest (ROIs)."""
    if detector is None or predictor is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0) # Use 0 for faster processing without upsampling
    if not faces:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, face)

    # Key landmark points for the 4 main regions
    forehead_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(19, 25)])
    left_cheek_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(1, 6)])
    right_cheek_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(12, 17)])
    philtrum_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in [32, 33, 34, 50, 51, 52]])

    roi_patches = []
    # Regions and the number of sub-ROIs to extract from each
    main_regions = [(forehead_pts, 6), (left_cheek_pts, 5), (right_cheek_pts, 5), (philtrum_pts, 6)]

    for pts, num_sub_rois in main_regions:
        x, y, w, h = cv2.boundingRect(pts)
        if w < 2 or h < 2: continue

        # Simple grid subdivision to extract sub-ROIs
        cols = 3 if num_sub_rois > 2 else 2
        rows = int(np.ceil(num_sub_rois / cols))
        sub_w, sub_h = max(1, w // cols), max(1, h // rows)

        for i in range(num_sub_rois):
            px = x + (i % cols) * sub_w
            py = y + (i // rows) * sub_h
            patch = frame[py:py+sub_h, px:px+sub_w]
            if patch.size > 0:
                roi_patches.append(patch)

    return roi_patches if len(roi_patches) == config.ROI_COUNT else None

def _get_mean_rgb_from_rois(rois: List[np.ndarray]) -> np.ndarray:
    """Calculates the spatial average of RGB values for each ROI patch."""
    return np.array([np.mean(roi, axis=(0, 1)) for roi in rois])

def _apply_filters_and_chrom(rgb_signals: np.ndarray) -> np.ndarray:
    """Applies detrending, filtering, and CHROM to extract the rPPG signal."""
    # 1. Sliding Window Detrending
    detrended_rgb = np.zeros_like(rgb_signals)
    window_size = 15
    for i in range(len(rgb_signals)):
        start = max(0, i - window_size // 2)
        end = min(len(rgb_signals), i + window_size // 2 + 1)
        detrended_rgb[i] = rgb_signals[i] - np.mean(rgb_signals[start:end], axis=0)

    # 2. Butterworth Bandpass Filter
    nyquist = 0.5 * config.VIDEO_FPS
    low = config.LOW_CUTOFF / nyquist
    high = config.HIGH_CUTOFF / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_rgb = filtfilt(b, a, detrended_rgb, axis=0)

    # 3. CHROM method
    r, g, b = filtered_rgb[:, 0], filtered_rgb[:, 1], filtered_rgb[:, 2]
    x_signal = r - g
    y_signal = 0.5 * (r + g) - b
    std_x = np.std(x_signal)
    std_y = np.std(y_signal)
    rppg_signal = x_signal - (std_x / std_y) * y_signal if std_y > 0 else x_signal

    # 4. Final normalization (z-score)
    mean, std = np.mean(rppg_signal), np.std(rppg_signal)
    return (rppg_signal - mean) / std if std > 0 else np.zeros_like(rppg_signal)

def create_mvhm(rppg_signals: np.ndarray, fft_signals: np.ndarray) -> np.ndarray:
    """
    --- CRITICAL FIX ---
    Creates a Matrix Visualization Heatmap (MVHM) by stacking independently
    normalized rPPG and FFT signal blocks.
    """
    # Normalize the rPPG block (top half) to 0-255 range
    min_rppg, max_rppg = np.min(rppg_signals), np.max(rppg_signals)
    norm_rppg = 255 * (rppg_signals - min_rppg) / (max_rppg - min_rppg) if (max_rppg - min_rppg) > 0 else np.zeros_like(rppg_signals)

    # Normalize the FFT block (bottom half) to 0-255 range
    min_fft, max_fft = np.min(fft_signals), np.max(fft_signals)
    norm_fft = 255 * (fft_signals - min_fft) / (max_fft - min_fft) if (max_fft - min_fft) > 0 else np.zeros_like(fft_signals)

    # Stack the independently normalized blocks
    combined_matrix = np.vstack((norm_rppg, norm_fft)).astype(np.uint8)

    # Resize to the target resolution and convert to 3-channel RGB
    heatmap_resized = cv2.resize(combined_matrix, config.MVHM_RESOLUTION, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(heatmap_resized, cv2.COLOR_GRAY2RGB)


def video_to_mvhm_from_frames(frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Core processing function: converts a list of frames into an MVHM image
    and returns intermediate signals for reuse.
    """
    if not frames: return None

    # Process frames to get RGB signals per frame
    rgb_signals_list = []
    last_known_rgb = None
    valid_frame_count = 0
    for frame in frames:
        rois = _extract_face_rois(frame)
        if rois:
            current_rgb = _get_mean_rgb_from_rois(rois)
            rgb_signals_list.append(current_rgb)
            last_known_rgb = current_rgb
            valid_frame_count += 1
        elif last_known_rgb is not None:
            rgb_signals_list.append(last_known_rgb) # Use last known if face is lost
        else:
            # If no face has been found yet, pad with zeros
            rgb_signals_list.append(np.zeros((config.ROI_COUNT, 3)))

    if valid_frame_count < config.MIN_VALID_FRAMES:
        logging.warning(f"Segment skipped: only {valid_frame_count}/{len(frames)} frames had a detectable face.")
        return None

    processed_rgb_signals = np.array(rgb_signals_list)

    # Generate rPPG signals for each ROI: (num_rois, num_frames)
    rppg_signals = np.array([_apply_filters_and_chrom(processed_rgb_signals[:, i, :]) for i in range(config.ROI_COUNT)])

    # Generate FFT signals from rPPG signals
    fft_signals = np.abs(fft(rppg_signals, axis=1))

    # Ensure shapes match before creating the MVHM
    assert rppg_signals.shape == fft_signals.shape

    # Create the final MVHM image
    mvhm_image = create_mvhm(rppg_signals, fft_signals)

    return {
        'mvhm_image': mvhm_image,
        'rppg_signals': rppg_signals,
        'fft_signals': fft_signals
    }

def get_inference_transform() -> transforms.Compose:
    """Returns the image transformation pipeline for inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MVHM_MEAN, std=config.MVHM_STD)
    ])

def process_video_for_inference(video_path: str):
    """
    Generator function for inference. Reads a video, breaks it into segments,
    processes each, and yields the results efficiently.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    transform = get_inference_transform()
    frame_buffer, start_time = [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_buffer.append(frame)

        # When buffer is full, process the segment
        if len(frame_buffer) == config.NUM_FRAMES:
            segment_duration = len(frame_buffer) / fps if fps > 0 else config.VIDEO_DURATION_SECS
            segment_data = {'start_time': start_time, 'end_time': start_time + segment_duration}

            processed_output = video_to_mvhm_from_frames(frame_buffer)

            if processed_output:
                mvhm_tensor = transform(processed_output['mvhm_image']).unsqueeze(0).to(config.DEVICE)
                avg_rppg = np.mean(processed_output['rppg_signals'], axis=0)
                avg_fft = np.mean(processed_output['fft_signals'], axis=0)

                segment_data.update({
                    'status': 'success',
                    'mvhm_tensor': mvhm_tensor,
                    'rppg_signal_avg': avg_rppg,
                    'fft_signal_avg': avg_fft
                })
            else:
                segment_data['status'] = 'fail'

            yield segment_data

            # Reset for the next segment
            start_time += segment_duration
            frame_buffer.clear()

    cap.release()