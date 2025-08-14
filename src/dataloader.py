# src/dataloader.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging
import numpy as np

from . import config

# --- Custom Augmentation Transforms ---

class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class SignalCutout(object):
    """
    Randomly erases N horizontal rows (signals) from the MVHM tensor.
    Note: This is an experimental and potentially destructive augmentation.
    """
    def __init__(self, n_holes, length=1):
        self.n_holes = n_holes
        self.length = length # Corresponds to erasing one full signal row

    def __call__(self, img):
        h = img.size(1) # Height
        w = img.size(2) # Width

        for _ in range(self.n_holes):
            # Erase a row from the top (rPPG) or bottom (FFT) half
            is_top_half = np.random.rand() > 0.5
            half_height = h // 2

            y = np.random.randint(0, half_height)
            if not is_top_half:
                y += half_height

            y1 = np.clip(y, 0, h)
            y2 = np.clip(y + self.length, 0, h)

            img[:, y1:y2, :] = 0. # Erase the row

        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(n_holes={self.n_holes}, length={self.length})'


class PreprocessedMVHMDataset(Dataset):
    """PyTorch Dataset that loads pre-generated MVHM images from disk."""
    def __init__(self, file_paths, labels, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

        # --- Use different transforms for training and validation ---
        if self.is_train:
            # Apply augmentations only to the training set
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Note: SignalCutout can be very destructive. Disable if results are poor.
                SignalCutout(n_holes=2, length=1),
                transforms.Normalize(mean=config.MVHM_MEAN, std=config.MVHM_STD),
                AddGaussianNoise(mean=0, std=0.05),
            ])
        else:
            # No augmentation for validation/testing
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MVHM_MEAN, std=config.MVHM_STD)
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Use Pillow to open and convert to RGB, which is standard
            image = Image.open(self.file_paths[idx]).convert("RGB")
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            transformed_image = self.transform(image)
            return transformed_image, label
        except Exception as e:
            logging.error(f"Error loading or transforming image {self.file_paths[idx]}: {e}")
            # Return None to be filtered by the collate_fn
            return None

def collate_fn(batch):
    """Custom collate function to filter out None values from the batch."""
    # Filter out samples that failed to load
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # If the whole batch failed, return empty tensors
        return torch.Tensor(), torch.Tensor()

    # Use the default collate function on the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)