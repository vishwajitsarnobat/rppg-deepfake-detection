# src/utils.py
import os
import logging
import requests
import bz2
from tqdm import tqdm

from . import config

def download_dlib_model():
    """Downloads and extracts the dlib face predictor model if it doesn't exist."""
    model_path = config.DLIB_MODEL_PATH
    if os.path.exists(model_path):
        return

    logging.info("Dlib face predictor model not found. Downloading...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    url = config.DLIB_MODEL_URL
    compressed_path = model_path + ".bz2"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(compressed_path, 'wb') as f, tqdm(
                desc="Downloading dlib model",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        logging.info("Extracting model...")
        with bz2.BZ2File(compressed_path, 'rb') as f_in, open(model_path, 'wb') as f_out:
            f_out.write(f_in.read())
        
        os.remove(compressed_path)
        logging.info("Dlib model downloaded and extracted successfully.")

    except Exception as e:
        logging.error(f"Failed to download or extract dlib model: {e}")
        logging.error("Please manually download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        logging.error(f"And place 'shape_predictor_68_face_landmarks.dat' in the '{config.MODEL_DIR}' directory.")
        exit(1)