# src/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns
import os

from . import config

def plot_confusion_matrix(cm, epoch_name, class_names=['Real', 'Fake']):
    """Creates a heatmap visualization of the confusion matrix and saves it."""
    output_dir = os.path.join(config.OUTPUT_DIR, "cm_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    
    ax.set_title(f'Confusion Matrix - {epoch_name}', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"confusion_matrix_{epoch_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def calculate_bpm(fft_signal, fps):
    """Calculates BPM from the peak frequency in the FFT signal within the valid range."""
    n = len(fft_signal)
    if n == 0: return 0
    
    freqs = np.fft.fftfreq(n, d=1.0 / fps)
    
    # Find indices corresponding to the valid heart rate frequency range
    valid_indices = np.where((freqs >= config.LOW_CUTOFF) & (freqs <= config.HIGH_CUTOFF))
    if len(valid_indices[0]) == 0: return 0
    
    # Find the peak frequency in the valid range
    peak_index_in_valid = np.argmax(np.abs(fft_signal[valid_indices]))
    peak_freq = freqs[valid_indices][peak_index_in_valid]
    
    return peak_freq * 60

def plot_rppg_signal(analysis_results, output_path, prediction):
    """Plots the extracted rPPG signal, colored by prediction."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 5))
    
    result = analysis_results[0]
    color = '#d62728' if prediction == 'Manipulated' else '#2ca02c'
    title = f'Extracted rPPG Signal (Predicted: {prediction})'

    if result.get('status') == 'success' and result.get('rppg_signal_avg') is not None:
        signal = result['rppg_signal_avg']
        time_axis = np.arange(len(signal)) / config.VIDEO_FPS
        ax.plot(time_axis, signal, color=color, linewidth=1.5, label=f'rPPG Signal ({prediction})')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Signal Amplitude', fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No rPPG signal could be extracted.', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"rPPG signal graph saved to {output_path}")

def plot_timeline(analysis_results, video_duration, output_path, prediction):
    """Creates a simple timeline visualization for a single video analysis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 2.5))
    
    result = analysis_results[0]
    
    # Determine color based on prediction status
    if result.get('status') == 'success':
        color = '#d62728' if prediction == 'Manipulated' else '#2ca02c'
        label = f'Processed as {prediction}'
    else:
        color = 'gray'
        label = 'Analysis Failed'
        
    ax.axvspan(0, video_duration, color=color, alpha=0.7)

    ax.set_xlim(0, max(video_duration, 1.0))
    ax.set_ylim(0, 1)
    ax.set_title('Video Analysis Timeline', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_yticks([])

    # Create a single patch for the legend
    patch = plt.Rectangle((0, 0), 1, 1, color=color, label=label)
    ax.legend(handles=[patch], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Prediction timeline saved to {output_path}")