# src/preprocess_data.py
import os
import cv2
from tqdm import tqdm
import logging
import random
import concurrent.futures
from typing import List, Tuple

from src.processor import video_to_mvhm_from_frames
from src import config

def collect_videos_from_dirs(list_of_dirs: List[str]) -> List[str]:
    """Walks through directories to collect all unique video files."""
    video_files = set()
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    for directory in list_of_dirs:
        if not os.path.exists(directory):
            logging.warning(f"Source directory not found, skipping: {directory}")
            continue
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(VIDEO_EXTENSIONS):
                    video_files.add(os.path.join(dirpath, filename))
    return list(video_files)

def process_and_save_worker(task_args: Tuple[str, str, str]) -> int:
    """
    Worker function to process one video and save the resulting MVHM image.
    Returns 1 if saved, 0 otherwise.
    """
    video_path, label, output_dir = task_args
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{label}_{base_video_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Skip if this video has already been processed
    if os.path.exists(output_path):
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return 0
    
    frames_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
    cap.release()

    if not frames_buffer:
        return 0

    processed_data = video_to_mvhm_from_frames(frames_buffer)
    if processed_data and 'mvhm_image' in processed_data:
        cv2.imwrite(output_path, processed_data['mvhm_image'])
        return 1
    
    return 0

def run_preprocessing():
    """
    Processes all videos in parallel, generating and saving one MVHM image for each.
    """
    logging.info("--- Starting Full Video Preprocessing (One MVHM per Video) ---")

    real_files = collect_videos_from_dirs(config.REAL_VIDEO_DIRS)
    fake_files = collect_videos_from_dirs(config.FAKE_VIDEO_DIRS)

    if not real_files and not fake_files:
        logging.error("No video files found in the specified directories. Aborting.")
        return
        
    logging.info(f"Found {len(real_files)} real videos and {len(fake_files)} fake videos.")
    
    tasks = []
    for label, files in [("real", real_files), ("fake", fake_files)]:
        output_dir = os.path.join(config.PREPROCESSED_DATA_DIR, label)
        os.makedirs(output_dir, exist_ok=True)
        for video_path in files:
            tasks.append((video_path, label, output_dir))
    
    random.shuffle(tasks)

    logging.info(f"Starting processing for {len(tasks)} videos.")
    total_saved = 0
    # Use fewer workers if CPU count is high to avoid memory issues with full video processing
    max_workers = min(os.cpu_count(), 8)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_and_save_worker, tasks), total=len(tasks), desc="Processing Videos"))
        total_saved = sum(results)

    logging.info(f"--- Preprocessing Complete ---")
    logging.info(f"Successfully processed and saved {total_saved} new MVHM images.")