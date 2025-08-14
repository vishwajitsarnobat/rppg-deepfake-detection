# src/config.py

import os
import torch

# --- DYNAMIC PATH CONFIGURATION ---
# These paths are relative to the project's root directory.
# This requires you to always run scripts from the project root.
BASE_DATA_PATH = "data_subset"

# --- USER-CONFIGURABLE VIDEO SOURCE DIRECTORIES ---
REAL_VIDEO_DIRS = [
    os.path.join(BASE_DATA_PATH, "real"),
]
FAKE_VIDEO_DIRS = [
    os.path.join(BASE_DATA_PATH, "fake"),
]

# --- GLOBAL PROJECT CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSED_DATA_DIR = "data_preprocessed"
MODEL_DIR = "models"
LOG_FILE = "app.log"
OUTPUT_DIR = "outputs"

# --- DLIB MODEL ---
# This path is now correct relative to the project root.
DLIB_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

# --- MODEL CHECKPOINT PATHS ---
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_deepfake_detector.pth")
LATEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "latest_checkpoint.pth")

# --- DATA & PREPROCESSING PARAMETERS ---
VIDEO_FPS = 25.0
VIDEO_DURATION_SECS = 3
NUM_FRAMES = int(VIDEO_FPS * VIDEO_DURATION_SECS)
MIN_VALID_FRAMES = 60
ROI_COUNT = 22
MVHM_RESOLUTION = (240, 240)

# --- SIGNAL PROCESSING PARAMETERS ---
LOW_CUTOFF = 0.8
HIGH_CUTOFF = 3.0

# --- MODEL & TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.5
TEST_SPLIT_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 10

# --- IMAGE NORMALIZATION (Standard for ImageNet-pretrained models) ---
MVHM_MEAN = [0.485, 0.456, 0.406]
MVHM_STD = [0.229, 0.224, 0.225]