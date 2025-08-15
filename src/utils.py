# src/utils.py
import os
import logging
import requests
import bz2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class FocalLoss(nn.Module):
    """
    Focal Loss for hard-to-classify examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def download_dlib_model():
    """Downloads and extracts the dlib face predictor model if it doesn't exist."""
    if os.path.exists(config.DLIB_MODEL_PATH): return
    logging.info("Dlib face predictor model not found. Downloading...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    try:
        with requests.get(config.DLIB_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            compressed_path = config.DLIB_MODEL_PATH + ".bz2"
            with open(compressed_path, 'wb') as f, tqdm(
                desc="Downloading dlib model", total=total_size, unit='iB', unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        logging.info("Extracting model...")
        with bz2.BZ2File(compressed_path, 'rb') as f_in, open(config.DLIB_MODEL_PATH, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(compressed_path)
        logging.info("Dlib model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download dlib model: {e}. Please download it manually.")
        exit(1)