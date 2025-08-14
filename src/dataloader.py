# src/dataloader.py
import os
import glob
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import config

def list_preprocessed_files(base_dir):
    """Finds all real/fake images in the preprocessed directory."""
    file_paths, labels = [], []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        data_dir = os.path.join(base_dir, label_name)
        if not os.path.isdir(data_dir):
            logging.warning(f"Preprocessed directory not found: {data_dir}")
            continue
        for ext in config.IMAGE_EXTS:
            found_files = glob.glob(os.path.join(data_dir, f"*{ext}"))
            file_paths.extend(found_files)
            labels.extend([label_id] * len(found_files))
    
    if not file_paths:
        logging.error(f"No preprocessed files found in {base_dir}. Please run preprocessing.")
    return file_paths, labels

class PreprocessedMVHMDataset(Dataset):
    """PyTorch Dataset for loading MVHM images with strong augmentations."""
    def __init__(self, file_paths, labels, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train
        self.transform = self._build_transforms()

    def _build_transforms(self):
        """Builds the transformation pipeline."""
        transform_list = [transforms.Resize(config.MVHM_RESOLUTION)]

        if self.is_train and config.USE_AUGMENTATION:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=config.AUG_H_FLIP_PROB),
                transforms.RandomRotation(config.AUG_ROTATION_DEGREES),
                transforms.ColorJitter(**config.AUG_COLOR_JITTER),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MVHM_MEAN, std=config.MVHM_STD)
        ])
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            image = Image.open(file_path).convert("RGB")
            image = self.transform(image)
            label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
            return image, label
        except Exception as e:
            logging.error(f"Failed to load or transform image {file_path}: {e}")
            return None # Will be filtered by collate_fn

def collate_fn(batch):
    """Filters out None values from a batch before collating."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)