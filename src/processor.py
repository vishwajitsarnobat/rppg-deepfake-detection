# src/processor.py
import cv2
import dlib
import numpy as np
import logging
import os
from torchvision import transforms
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from typing import List, Dict, Any, Optional

from . import config

# --- dlib Model Loading ---
try:
    if not os.path.exists(config.DLIB_MODEL_PATH):
        raise FileNotFoundError(f"Dlib model not found at {config.DLIB_MODEL_PATH}. Please run the download script or place it in the 'models' directory.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.DLIB_MODEL_PATH)
except (FileNotFoundError, RuntimeError) as e:
    logging.error(f"FATAL: Failed to initialize Dlib. {e}")
    detector = None
    predictor = None

def _extract_face_rois(frame: np.ndarray) -> Optional[List[np.ndarray]]:
    """Detects the largest face and extracts 22 specific Regions of Interest (ROIs)."""
    if detector is None or predictor is None: return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if not faces: return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, face)

    # Define ROIs based on dlib's 68 landmarks
    forehead_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(19, 25)])
    left_cheek_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(1, 6)])
    right_cheek_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(12, 17)])
    philtrum_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in [32, 33, 34, 50, 51, 52]])

    roi_patches = []
    main_regions = [(forehead_pts, 6), (left_cheek_pts, 5), (right_cheek_pts, 5), (philtrum_pts, 6)]

    for pts, num_sub_rois in main_regions:
        x, y, w, h = cv2.boundingRect(pts)
        if w < 2 or h < 2: continue
        
        # Subdivide the main ROI into smaller patches
        cols = 3 if num_sub_rois > 2 else 2
        rows = int(np.ceil(num_sub_rois / cols))
        sub_w, sub_h = max(1, w // cols), max(1, h // rows)

        for i in range(num_sub_rois):
            px, py = x + (i % cols) * sub_w, y + (i // rows) * sub_h
            patch = frame[py:py+sub_h, px:px+sub_w]
            if patch.size > 0:
                roi_patches.append(patch)

    return roi_patches if len(roi_patches) == config.ROI_COUNT else None

def _get_mean_rgb_from_rois(rois: List[np.ndarray]) -> np.ndarray:
    """Calculates the spatial average of RGB values for each ROI patch."""
    return np.array([np.mean(roi, axis=(0, 1)) for roi in rois])

def _apply_filters_and_chrom(rgb_signals: np.ndarray) -> np.ndarray:
    """
    Applies detrending, filtering, and CHROM to extract the rPPG signal.
    Normalization is handled later during MVHM creation to preserve features.
    """
    # 1. Detrending (simple linear detrending is often enough)
    detrended_rgb = rgb_signals - np.mean(rgb_signals, axis=0)

    # 2. Butterworth Bandpass Filter
    nyquist = 0.5 * config.VIDEO_FPS
    low, high = config.LOW_CUTOFF / nyquist, config.HIGH_CUTOFF / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_rgb = filtfilt(b, a, detrended_rgb, axis=0)

    # 3. CHROM method for rPPG signal extraction
    r, g, b = filtered_rgb[:, 0], filtered_rgb[:, 1], filtered_rgb[:, 2]
    x_signal = r - g
    y_signal = 0.5 * (r + g) - b
    std_x, std_y = np.std(x_signal), np.std(y_signal)
    
    # Calculate alpha to avoid division by zero
    alpha = std_x / std_y if std_y > 0 else 1.0
    rppg_signal = x_signal - alpha * y_signal
    
    return rppg_signal

def create_mvhm(rppg_signals: np.ndarray, fft_signals: np.ndarray) -> np.ndarray:
    """
    Creates a Matrix Visualization Heatmap (MVHM) by stacking independently
    normalized rPPG and FFT signal blocks. This is the correct place for normalization.
    """
    # Normalize the rPPG block (top half) to 0-255 range.
    min_rppg, max_rppg = np.min(rppg_signals), np.max(rppg_signals)
    if (max_rppg - min_rppg) > 1e-6:
        norm_rppg = 255 * (rppg_signals - min_rppg) / (max_rppg - min_rppg)
    else:
        norm_rppg = np.zeros_like(rppg_signals)

    # Normalize the FFT block (bottom half)
    min_fft, max_fft = np.min(fft_signals), np.max(fft_signals)
    if (max_fft - min_fft) > 1e-6:
        norm_fft = 255 * (fft_signals - min_fft) / (max_fft - min_fft)
    else:
        norm_fft = np.zeros_like(fft_signals)

    # Combine and resize
    combined_matrix = np.vstack((norm_rppg, norm_fft)).astype(np.uint8)
    heatmap_resized = cv2.resize(combined_matrix, config.MVHM_RESOLUTION, interpolation=cv2.INTER_CUBIC)
    
    # Convert to 3-channel RGB for the model
    return cv2.cvtColor(heatmap_resized, cv2.COLOR_GRAY2RGB)

def video_to_mvhm_from_frames(frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Core processing function: converts a list of frames from a full video into one MVHM."""
    if not frames:
        logging.warning("Cannot process an empty list of frames.")
        return None

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
            # If face is lost, carry over the last known RGB values
            rgb_signals_list.append(last_known_rgb)
        else:
            # If no face has been detected yet, append zeros
            rgb_signals_list.append(np.zeros((config.ROI_COUNT, 3)))
    
    # Check if the video had enough valid frames to be useful
    if valid_frame_count < config.MIN_VALID_FRAMES_FOR_SEGMENT:
        logging.warning(f"Video skipped: only {valid_frame_count}/{len(frames)} frames had a detectable face.")
        return None

    # Process all signals together
    processed_rgb_signals = np.array(rgb_signals_list)
    
    # Apply processing to each ROI's signal over time
    rppg_signals = np.array([_apply_filters_and_chrom(processed_rgb_signals[:, i, :]) for i in range(config.ROI_COUNT)])
    
    # Compute FFT for each ROI's rPPG signal
    fft_signals = np.abs(fft(rppg_signals, axis=1))

    # Ensure shapes are consistent
    assert rppg_signals.shape == fft_signals.shape, "rPPG and FFT signal shapes must match"

    mvhm_image = create_mvhm(rppg_signals, fft_signals)

    return {
        'mvhm_image': mvhm_image,
        'rppg_signals': rppg_signals, # For visualization
        'fft_signals': fft_signals   # For visualization
    }

def get_inference_transform() -> transforms.Compose:
    """Returns the image transformation pipeline for inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MVHM_MEAN, std=config.MVHM_STD)
    ])

def process_video_for_inference(video_path: str):
    """
    Generator that processes a full video and yields a single result dictionary.
    This replaces the segmented approach.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    frame_buffer = []
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
    
    cap.release()
    
    if not frame_buffer:
        logging.error(f"Video file is empty or could not be read: {video_path}")
        return

    video_duration = len(frame_buffer) / fps
    result_data = {'start_time': 0, 'end_time': video_duration, 'status': 'fail'}
    
    processed_output = video_to_mvhm_from_frames(frame_buffer)
    transform = get_inference_transform()

    if processed_output:
        result_data.update({
            'status': 'success',
            'mvhm_tensor': transform(processed_output['mvhm_image']).unsqueeze(0).to(config.DEVICE),
            'rppg_signal_avg': np.mean(processed_output['rppg_signals'], axis=0),
            'fft_signal_avg': np.mean(processed_output['fft_signals'], axis=0),
        })
    
    yield result_data