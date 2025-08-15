# src/config.py
import os
import torch

# --- PATHS ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_SUBSET_DIR = os.path.join(PROJECT_ROOT, "data_subset")
PREPROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_preprocessed")
REAL_VIDEO_DIRS = [os.path.join(DATA_SUBSET_DIR, "real")]
FAKE_VIDEO_DIRS = [os.path.join(DATA_SUBSET_DIR, "fake")]

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")

DLIB_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_deepfake_detector.pth")
LATEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "latest_checkpoint.pth")

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA & PREPROCESSING ---
VIDEO_FPS = 25.0
SEGMENT_DURATION_SECS = 3
NUM_FRAMES_PER_SEGMENT = int(VIDEO_FPS * SEGMENT_DURATION_SECS)
MIN_VALID_FRAMES_FOR_SEGMENT = 60 # A segment is only valid if it has this many face detections
ROI_COUNT = 22
MVHM_RESOLUTION = (240, 240)
IMAGE_EXTS = ['.png', '.jpg', '.jpeg']

# Heart Rate Estimation
LOW_CUTOFF = 0.8  # Corresponds to 48 BPM
HIGH_CUTOFF = 3.0 # Corresponds to 180 BPM

# --- DATALOADER ---
BATCH_SIZE = 16
NUM_WORKERS = min(os.cpu_count(), 4)
PIN_MEMORY = DEVICE.type == "cuda"

# --- MODEL & TRAINING ---
# Hyperparameters
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4

# Two-Stage Training Strategy
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
FINETUNE_EPOCH = 5
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7

# Augmentation
USE_AUGMENTATION = True
AUG_H_FLIP_PROB = 0.5
AUG_ROTATION_DEGREES = 10
AUG_COLOR_JITTER = {"brightness": 0.1, "contrast": 0.1}

# Normalization for MVHM images (ImageNet stats)
MVHM_MEAN = [0.485, 0.456, 0.406]
MVHM_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42