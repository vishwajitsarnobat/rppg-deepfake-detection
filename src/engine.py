# src/engine.py
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import cv2

from . import config
from .dataloader import list_preprocessed_files, PreprocessedMVHMDataset, collate_fn
from .model import DeepfakeDetector
from .processor import process_video_for_inference
from .visualizer import plot_confusion_matrix, plot_timeline, plot_rppg_signal, calculate_bpm

class EarlyStopper:
    """Simple early stopping utility."""
    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Indicates improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given data loader."""
    model.eval()
    all_labels, all_preds, all_probs, all_losses = [], [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            if inputs.numel() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Use AMP for faster inference if available
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_losses.append(loss.item())
            
    val_loss = np.mean(all_losses) if all_losses else 0
    val_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    val_auc = roc_auc_score(all_labels, all_probs) if all_labels else 0
    cm = confusion_matrix(all_labels, all_preds) if all_labels else None
    
    return val_loss, val_acc, val_auc, cm

def run_training(resume=False):
    """Main training loop implementing the two-stage strategy."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    file_paths, labels = list_preprocessed_files(config.PREPROCESSED_DATA_DIR)
    if not file_paths:
        logging.error("No preprocessed files found. Run 'preprocess' first.")
        return

    # Create train/validation splits
    train_fp, val_fp, train_l, val_l = train_test_split(
        file_paths, labels, test_size=0.2, random_state=config.RANDOM_SEED, stratify=labels
    )
    train_ds = PreprocessedMVHMDataset(train_fp, train_l, is_train=True)
    val_ds = PreprocessedMVHMDataset(val_fp, val_l, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)

    device = config.DEVICE
    model = DeepfakeDetector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss() # Better for this architecture
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    logging.info(f"--- Starting Training on {device} ---")
    logging.info(f"Dataset: {len(train_ds)} training, {len(val_ds)} validation images.")

    # --- Stage 1: Train Classifier Head Only ---
    logging.info(f"--- Stage 1: Training Head ({config.FINETUNE_EPOCH} epochs) ---")
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config.LEARNING_RATE_HEAD, weight_decay=config.WEIGHT_DECAY)
    
    for epoch in range(1, config.FINETUNE_EPOCH + 1):
        model.train()
        model.features.eval() # Keep backbone frozen
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}/{config.FINETUNE_EPOCH}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
        
        val_loss, val_acc, val_auc, cm = evaluate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        if cm is not None: plot_confusion_matrix(cm, f"stage1_epoch_{epoch}")

    # --- Stage 2: Fine-Tune Full Model ---
    logging.info(f"--- Stage 2: Fine-Tuning Full Model (Max {config.MAX_EPOCHS} epochs) ---")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_FINETUNE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    early_stopper = EarlyStopper()

    for epoch in range(config.FINETUNE_EPOCH + 1, config.MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/{config.MAX_EPOCHS}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
        
        val_loss, val_acc, val_auc, cm = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        if cm is not None: plot_confusion_matrix(cm, f"stage2_epoch_{epoch}")

        if early_stopper(val_loss):
            logging.info(f"Validation loss improved to {val_loss:.4f}. Saving best model.")
            torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)
        
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("--- Training Finished ---")
    # Load the best model for a final evaluation
    model.load_state_dict(torch.load(config.BEST_MODEL_SAVE_PATH))
    run_evaluation(model)


def run_evaluation(model=None):
    """Evaluates the best model on the validation set."""
    logging.info("--- Starting Final Evaluation ---")
    if model is None:
        model = DeepfakeDetector(pretrained=False).to(config.DEVICE)
        model.load_state_dict(torch.load(config.BEST_MODEL_SAVE_PATH, map_location=config.DEVICE))
    
    file_paths, labels = list_preprocessed_files(config.PREPROCESSED_DATA_DIR)
    _, val_fp, _, val_l = train_test_split(
        file_paths, labels, test_size=0.2, random_state=config.RANDOM_SEED, stratify=labels
    )
    val_ds = PreprocessedMVHMDataset(val_fp, val_l, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    
    val_loss, val_acc, val_auc, cm = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), config.DEVICE)
    logging.info(f"Final Evaluation | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
    if cm is not None: plot_confusion_matrix(cm, "final_evaluation")

def run_prediction(video_path):
    """Runs a full prediction on a single video file."""
    logging.info(f"--- Starting Prediction for: {os.path.basename(video_path)} ---")
    model = DeepfakeDetector(pretrained=False).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.BEST_MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        logging.error("Model file not found. Please train the model first using the 'train' command.")
        return
        
    model.eval()

    # process_video_for_inference is now a generator that yields one result
    analysis_results = list(process_video_for_inference(video_path))

    if not analysis_results:
        logging.error("Could not process the video. It might be too short, corrupted, or no face was detected.")
        return

    result_data = analysis_results[0]
    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    if result_data['status'] == 'success':
        with torch.no_grad():
            output = model(result_data['mvhm_tensor'])
            confidence = torch.sigmoid(output).item()
        
        prediction = "Manipulated" if confidence > 0.5 else "Authentic"
        overall_confidence = confidence if prediction == "Manipulated" else 1 - confidence
        
        logging.info(f"\n--- PREDICTION REPORT ---")
        logging.info(f"Overall Result: {prediction} (Confidence: {overall_confidence:.2%})")

        # Visualization
        output_dir = config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        plot_timeline([result_data], result_data['end_time'], os.path.join(output_dir, f"{base_filename}_timeline.png"), prediction)
        plot_rppg_signal([result_data], os.path.join(output_dir, f"{base_filename}_rppg_signal.png"), prediction)
        
        # Calculate and log BPM
        bpm = calculate_bpm(result_data['fft_signal_avg'], config.VIDEO_FPS)
        logging.info(f"Estimated Heart Rate from Authentic Signal: {bpm:.2f} BPM")
    else:
        logging.error("Analysis failed. No valid MVHM could be generated.")
        # Still plot the timeline to show failure
        plot_timeline([result_data], 1, os.path.join(config.OUTPUT_DIR, f"{base_filename}_timeline.png"), "Failed")