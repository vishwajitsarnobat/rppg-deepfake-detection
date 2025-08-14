# src/preprocess_data.py

import os
import cv2
from tqdm import tqdm
import logging
import random
import concurrent.futures

from src.processor import video_to_mvhm_from_frames  # noqa: E402
from src import config  # noqa: E402

def collect_videos_from_dirs(list_of_dirs):
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

def process_and_save_worker(task_args):
    """
    A single unit of work for a process pool. It opens a video, extracts the
    first 3-second segment, processes it into an MVHM, and saves it.
    """
    video_path, output_path = task_args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return False

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    # Read just enough frames for one segment
    for _ in range(config.NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        # Optional: resize frame here if videos are very high-res to speed up face detection
        frames.append(frame)
    cap.release()

    if len(frames) < config.MIN_VALID_FRAMES:
        logging.warning(f"Skipping {os.path.basename(video_path)}: not enough frames ({len(frames)}).")
        return False

    # This function now returns a dictionary
    processed_data = video_to_mvhm_from_frames(frames)
    if processed_data and 'mvhm_image' in processed_data:
        cv2.imwrite(output_path, processed_data['mvhm_image'])
        return True
    return False

def run_preprocessing():
    """Processes videos in parallel, intelligently skipping already processed files."""
    logging.info("--- Starting Offline Data Preprocessing ---")

    real_files = collect_videos_from_dirs(config.REAL_VIDEO_DIRS)
    fake_files = collect_videos_from_dirs(config.FAKE_VIDEO_DIRS)

    if not real_files and not fake_files:
        logging.error("No video files found in the directories specified in config.py. Aborting.")
        return

    logging.info(f"Found {len(real_files)} unique real videos and {len(fake_files)} unique fake videos.")
    all_videos = [('real', path) for path in real_files] + [('fake', path) for path in fake_files]
    random.shuffle(all_videos)

    real_output_dir = os.path.join(config.PREPROCESSED_DATA_DIR, 'real')
    fake_output_dir = os.path.join(config.PREPROCESSED_DATA_DIR, 'fake')
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)

    tasks_to_run = []
    for label, video_path in all_videos:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{label}_{video_id}.png"
        output_path = os.path.join(config.PREPROCESSED_DATA_DIR, label, output_filename)

        if not os.path.exists(output_path):
            tasks_to_run.append((video_path, output_path))

    if not tasks_to_run:
        logging.info("All videos have already been preprocessed. Nothing to do.")
        return

    logging.info(f"Skipping {len(all_videos) - len(tasks_to_run)} already processed videos. Processing {len(tasks_to_run)} new videos.")
    num_processed = 0
    max_workers = max(1, os.cpu_count() - 2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_and_save_worker, tasks_to_run), total=len(tasks_to_run), desc="Preprocessing Videos"))
        num_processed = sum(results)

    logging.info(f"--- Preprocessing Complete --- \nSuccessfully processed and saved {num_processed} new MVHM images.")