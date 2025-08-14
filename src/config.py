# src/config.py
import os
import torch

# --- PATHS ---
# Robustly define paths relative to this file's location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# --- DATA PATHS ---
DATA_SUBSET_DIR = os.path.join(PROJECT_ROOT, "data_subset")
PREPROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_preprocessed")
REAL_VIDEO_DIRS = [os.path.join(DATA_SUBSET_DIR, "real")]
FAKE_VIDEO_DIRS = [os.path.join(DATA_SUBSET_DIR, "fake")]

# --- OUTPUT PATHS ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CM_PLOTS_DIR = os.path.join(OUTPUT_DIR, "cm_plots")
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")

# --- DLIB MODEL ---
DLIB_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA & PREPROCESSING ---
VIDEO_FPS = 25.0
VIDEO_DURATION_SECS = 3
NUM_FRAMES = int(VIDEO_FPS * VIDEO_DURATION_SECS)
MIN_VALID_FRAMES = 60
ROI_COUNT = 22
MVHM_RESOLUTION = (240, 240)

# --- DATALOADER ---
IMAGE_EXTS = [".png", ".jpg", ".jpeg"]
BATCH_SIZE = 16
NUM_WORKERS = min(os.cpu_count(), 4)
PIN_MEMORY = DEVICE.type == "cuda"

# --- MODEL & TRAINING HYPERPARAMETERS ---
# Regularization
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 5e-4       # AdamW is better with weight decay
LABEL_SMOOTHING = 0.1   # A common, effective value

# Two-Stage Training
LEARNING_RATE_HEAD = 5e-4
LEARNING_RATE_FINETUNE = 1e-5
FINETUNE_EPOCH = 5
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
# Unfreeze only the last 4 modules of VGG19 (safer)
UNFREEZE_LAST_N_MODULES = 4

# Augmentation
USE_AUGMENTATION = True
AUG_ROTATION_DEGREES = 7
AUG_H_FLIP_PROB = 0.5
AUG_COLOR_JITTER = {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05}

# Normalization
MVHM_MEAN = [0.485, 0.456, 0.406]
MVHM_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42