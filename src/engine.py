# src/engine.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import logging
from tqdm import tqdm
import numpy as np
import cv2

from . import config
from .dataloader import PreprocessedMVHMDataset, collate_fn
from .model import DeepfakeDetector
from .processor import process_video_for_inference
from .visualizer import plot_rppg_signal, plot_timeline, calculate_bpm

class EarlyStopper:
    """A simple class to handle early stopping."""
    def __init__(self, patience=1, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                self.early_stop = True

def run_training(resume=False):
    """Main function to run the model training process."""
    logging.info(f"--- Starting Model Training on {config.DEVICE} ---")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # --- 1. Dataset and Dataloader Setup ---
    preprocessed_dir = config.PREPROCESSED_DATA_DIR
    real_path = os.path.join(preprocessed_dir, 'real')
    fake_path = os.path.join(preprocessed_dir, 'fake')
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.png')]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith('.png')]

    if not real_files or not fake_files:
        logging.error("Training data not found. Please run the 'preprocess' command first.")
        return

    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=config.TEST_SPLIT_SIZE, random_state=42, stratify=labels)
    train_dataset = PreprocessedMVHMDataset(train_files, train_labels, is_train=True)
    val_dataset = PreprocessedMVHMDataset(val_files, val_labels, is_train=False)

    use_pin_memory = config.DEVICE.type == 'cuda'
    num_workers = min(os.cpu_count(), 4)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=use_pin_memory)
    logging.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- 2. Model, Optimizer, and Loss Function ---
    model = DeepfakeDetector(pretrained=not resume).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    num_real = sum(1 for label in train_labels if label == 0)
    num_fake = sum(1 for label in train_labels if label == 1)
    pos_weight = torch.tensor([num_real / num_fake if num_fake > 0 else 1.0], device=config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logging.info(f"Class imbalance weights: Real={num_real}, Fake={num_fake}. Using pos_weight: {pos_weight.item():.2f}")

    # --- FIX: Removed the 'verbose' argument which is deprecated/removed in newer PyTorch versions ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_pin_memory)
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE)
    start_epoch, best_val_acc = 0, 0.0

    # --- 3. Resume from Checkpoint if specified ---
    if resume and os.path.exists(config.LATEST_MODEL_SAVE_PATH):
        logging.info(f"Resuming training from checkpoint: {config.LATEST_MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.LATEST_MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logging.info(f"Resumed from Epoch {start_epoch}. Best validation accuracy so far: {best_val_acc:.4f}")

    # --- 4. Training Loop ---
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)

        for inputs, labels in progress_bar:
            if not inputs.nelement(): continue
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=use_pin_memory):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader) if train_loader else 0
        val_acc, val_loss, _, val_cm = run_evaluation(model, val_loader, criterion, is_standalone=False)

        logging.info(f"Epoch {epoch+1} Summary -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_cm is not None: logging.info(f"Validation Confusion Matrix:\n{val_cm}")
        
        # This will now print a message when the LR is reduced, providing the same functionality.
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            logging.info(f"Learning rate reduced from {old_lr} to {new_lr}")


        # --- 5. Save Checkpoints and Handle Early Stopping ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, config.LATEST_MODEL_SAVE_PATH)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"New best model found! Val Acc: {best_val_acc:.4f}. Saving to {config.BEST_MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)

        early_stopper(val_acc)
        if early_stopper.early_stop:
            break

    logging.info("--- Training Finished ---")


def run_evaluation(model=None, data_loader=None, criterion=None, is_standalone=True, model_type='best'):
    """Evaluates the model and returns performance metrics."""
    if is_standalone:
        logging.info(f"--- Starting Standalone Evaluation on '{model_type}' Model ---")
        model_path = config.BEST_MODEL_SAVE_PATH if model_type == 'best' else config.LATEST_MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            logging.error(f"Model not found at '{model_path}'. Please train first.")
            return 0, 0, 0, None

        model = DeepfakeDetector(pretrained=False).to(config.DEVICE)
        state_dict = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(state_dict.get('model_state_dict', state_dict))

        real_files = [os.path.join(config.PREPROCESSED_DATA_DIR, 'real', f) for f in os.listdir(os.path.join(config.PREPROCESSED_DATA_DIR, 'real'))]
        fake_files = [os.path.join(config.PREPROCESSED_DATA_DIR, 'fake', f) for f in os.listdir(os.path.join(config.PREPROCESSED_DATA_DIR, 'fake'))]
        dataset = PreprocessedMVHMDataset(real_files + fake_files, [0]*len(real_files) + [1]*len(fake_files))
        data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(os.cpu_count(), 4), collate_fn=collate_fn)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    all_preds_proba, all_labels = [], []
    eval_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=is_standalone, disable=not is_standalone):
            if not inputs.nelement(): continue
            inputs, labels_tensor = inputs.to(config.DEVICE), labels.to(config.DEVICE).view(-1,1)

            with torch.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type=='cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels_tensor)

            eval_loss += loss.item()
            all_preds_proba.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    if not all_labels: return 0.0, 0.0, 0.0, None
    avg_eval_loss = eval_loss / len(data_loader) if data_loader else 0
    all_labels, all_preds_proba = np.array(all_labels), np.array(all_preds_proba)
    pred_labels = (all_preds_proba > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds_proba) if len(np.unique(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, pred_labels)

    if not is_standalone:
        return accuracy, avg_eval_loss, roc_auc, cm

    logging.info(f"\n--- EVALUATION REPORT ---\nModel: {model_type}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nROC-AUC:  {roc_auc:.4f}\nConfusion Matrix (Real=0, Fake=1):\n{cm}\n-----------------------\n")
    return accuracy, avg_eval_loss, roc_auc, cm


def run_prediction(video_path: str, model_type='best'):
    """Runs a comprehensive, segmented analysis on a single video file."""
    logging.info(f"--- Starting Advanced Prediction for: {os.path.basename(video_path)} ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model_path = config.BEST_MODEL_SAVE_PATH if model_type == 'best' else config.LATEST_MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        logging.error(f"Model not found at '{model_path}'. Please train or specify a valid model type.")
        return

    model = DeepfakeDetector(pretrained=False).to(config.DEVICE)
    state_dict = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict.get('model_state_dict', state_dict))
    model.eval()

    segment_results = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps if fps > 0 else 0
    cap.release()

    with torch.no_grad():
        for segment_data in tqdm(process_video_for_inference(video_path), desc="Analyzing Video Segments", unit="segment"):
            if segment_data['status'] == 'success':
                output = model(segment_data['mvhm_tensor'])
                confidence = torch.sigmoid(output).item()
                prediction = "Fake" if confidence > 0.5 else "Real"
                final_confidence = confidence if prediction == "Fake" else 1 - confidence

                segment_data.update({
                    'prediction': prediction,
                    'confidence': final_confidence
                })
            segment_results.append(segment_data)

    if not segment_results:
        logging.error("Could not process any segments from the video. It might be too short or no faces were detected.")
        return

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    timeline_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}_timeline.png")
    rppg_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}_rppg_signal.png")

    plot_timeline(segment_results, video_duration, output_path=timeline_path)
    plot_rppg_signal(segment_results, output_path=rppg_path)

    manipulated_segments = [s for s in segment_results if s.get('prediction') == 'Fake']
    if manipulated_segments:
        overall_prediction = "Manipulated"
        overall_confidence = np.mean([s['confidence'] for s in manipulated_segments])
    else:
        overall_prediction = "Authentic"
        real_segments_conf = [s['confidence'] for s in segment_results if s.get('prediction') == 'Real']
        overall_confidence = np.mean(real_segments_conf) if real_segments_conf else 1.0

    logging.info(f"\n\n--- PREDICTION REPORT for {os.path.basename(video_path)} ---\n")
    logging.info(f"Overall Result: {overall_prediction} (Confidence: {overall_confidence:.2%})")
    logging.info(f"Analysis reports saved in: {config.OUTPUT_DIR}")

    if manipulated_segments:
        logging.info("\nDetected Manipulated Time Segments:")
        for seg in manipulated_segments:
            logging.info(f"  - {seg['start_time']:.2f}s to {seg['end_time']:.2f}s (Confidence of being fake: {seg['confidence']:.2%})")

    real_segments = [s for s in segment_results if s.get('prediction') == 'Real' and s.get('fft_signal_avg') is not None]
    if real_segments:
        valid_bpms = [bpm for bpm in [calculate_bpm(s['fft_signal_avg'], fps) for s in real_segments] if bpm > 0]
        if valid_bpms:
            avg_bpm = sum(valid_bpms) / len(valid_bpms)
            logging.info(f"\nEstimated Heart Rate (from authentic parts): {avg_bpm:.1f} BPM")

    logging.info("\n-----------------------------------------------------\n")