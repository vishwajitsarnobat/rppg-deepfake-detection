# src/engine.py
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from . import config
from .dataloader import list_preprocessed_files, PreprocessedMVHMDataset, collate_fn
from .model import DeepfakeDetector
from .visualizer import plot_confusion_matrix

class EarlyStopper:
    """Stops training when validation loss stops improving."""
    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True # Improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False # No improvement

def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_labels, all_preds, all_losses = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            if inputs.numel() == 0: 
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_losses.append(loss.item())
            
    val_loss = np.mean(all_losses) if all_losses else 0
    val_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    return val_loss, val_acc, all_labels, all_preds

def run_training():
    """Main training loop with two-stage fine-tuning."""
    # --- 1. Setup ---
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    file_paths, labels = list_preprocessed_files(config.PREPROCESSED_DATA_DIR)
    if not file_paths: 
        return

    train_fp, val_fp, train_l, val_l = train_test_split(
        file_paths, labels, test_size=0.2, random_state=config.RANDOM_SEED, stratify=labels
    )
    train_ds = PreprocessedMVHMDataset(train_fp, train_l, is_train=True)
    val_ds = PreprocessedMVHMDataset(val_fp, val_l, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)

    device = config.DEVICE
    model = DeepfakeDetector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    logging.info(f"--- Starting Training on {device} ---")
    logging.info(f"Dataset: {len(train_ds)} training, {len(val_ds)} validation samples.")

    # --- 2. Stage 1: Train Classifier Head ---
    logging.info(f"--- Stage 1: Training Head ({config.FINETUNE_EPOCH} epochs) ---")
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config.LEARNING_RATE_HEAD, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCH, eta_min=1e-7)

    for epoch in range(1, config.FINETUNE_EPOCH + 1):
        model.train()
        model.features.eval() # Keep backbone frozen
        
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}", leave=False)
        for inputs, labels in progress_bar:
            if inputs.numel() == 0: 
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Label smoothing
            if config.LABEL_SMOOTHING > 0:
                labels = labels * (1.0 - config.LABEL_SMOOTHING) + 0.5 * config.LABEL_SMOOTHING

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
        
        scheduler.step()
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        plot_confusion_matrix(confusion_matrix(y_true, y_pred), epoch)

    # --- 3. Stage 2: Fine-Tune Backbone ---
    logging.info(f"--- Stage 2: Fine-Tuning Backbone (last {config.UNFREEZE_LAST_N_MODULES} modules) ---")
    model.unfreeze_last_n_modules(config.UNFREEZE_LAST_N_MODULES)
    
    # Create new optimizer for all trainable parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE_FINETUNE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS - config.FINETUNE_EPOCH, eta_min=1e-8)
    early_stopper = EarlyStopper()

    for epoch in range(config.FINETUNE_EPOCH + 1, config.MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}", leave=False)
        for inputs, labels in progress_bar:
            if inputs.numel() == 0: 
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            if config.LABEL_SMOOTHING > 0:
                labels = labels * (1.0 - config.LABEL_SMOOTHING) + 0.5 * config.LABEL_SMOOTHING

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
        
        scheduler.step()
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        plot_confusion_matrix(confusion_matrix(y_true, y_pred), epoch)

        if early_stopper(val_loss):
            logging.info(f"Validation loss improved to {val_loss:.4f}. Saving best model.")
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "best_deepfake_detector.pth"))
        
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("--- Training Finished ---")

# The evaluation and prediction functions can be simplified or removed if not needed
# for the main training script, but are kept here for completeness.
def run_evaluation(model_path=None):
    pass # Implementation can be added back if needed

def run_prediction(video_path, model_type):
    pass # Implementation can be added back if needed