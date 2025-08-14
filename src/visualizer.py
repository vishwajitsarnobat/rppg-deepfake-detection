# src/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import logging

from . import config

def calculate_bpm(fft_signal, fps):
    """Calculates BPM from the peak frequency in the FFT signal within the valid range."""
    n = len(fft_signal)
    if n == 0: return 0

    freqs = np.fft.fftfreq(n, d=1.0 / fps)

    valid_indices = np.where((freqs >= config.LOW_CUTOFF) & (freqs <= config.HIGH_CUTOFF))
    if len(valid_indices[0]) == 0: return 0

    # Find the peak frequency in the valid heart rate range
    peak_index_in_valid = np.argmax(np.abs(fft_signal[valid_indices]))
    peak_freq = freqs[valid_indices][peak_index_in_valid]

    return peak_freq * 60

def plot_rppg_signal(segment_results, output_path="prediction_rppg_graph.png"):
    """Plots the concatenated rPPG signal from all 'Real' (authentic) segments."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 5))

    authentic_signals = [res['rppg_signal_avg'] for res in segment_results if res.get('prediction') == 'Real' and res.get('rppg_signal_avg') is not None]

    if not authentic_signals:
        ax.text(0.5, 0.5, 'No authentic rPPG signal was detected to plot.',
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Extracted rPPG Signal', fontsize=16)
    else:
        full_signal = np.concatenate(authentic_signals)
        time_axis = np.arange(len(full_signal)) / config.VIDEO_FPS
        ax.plot(time_axis, full_signal, color='#2ca02c', linewidth=1.5, label='rPPG Signal')
        ax.set_title('Extracted rPPG Signal from Authentic Segments', fontsize=16)
        ax.set_xlabel('Time (seconds of authentic footage)', fontsize=12)
        ax.set_ylabel('Normalized Signal Amplitude', fontsize=12)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"rPPG signal graph saved to {output_path}")

def plot_timeline(segment_results, video_duration, output_path="prediction_timeline.png"):
    """Creates a color-coded timeline visualization of the video analysis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 2.5))

    if not segment_results:
        ax.text(0.5, 0.5, 'No segments were analyzed.', ha='center', va='center', transform=ax.transAxes)
    else:
        for result in segment_results:
            color = 'gray'  # Default for failed segments
            if result.get('status') == 'success':
                color = '#d62728' if result.get('prediction') == 'Fake' else '#2ca02c'
            ax.axvspan(result['start_time'], result['end_time'], color=color, alpha=0.7)

    ax.set_xlim(0, max(video_duration, 1.0))
    ax.set_ylim(0, 1)
    ax.set_title('Video Manipulation Timeline', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_yticks([])

    patches = [
        plt.Rectangle((0, 0), 1, 1, color='#2ca02c', label='Authentic Segment'),
        plt.Rectangle((0, 0), 1, 1, color='#d62728', label='Manipulated Segment'),
        plt.Rectangle((0, 0), 1, 1, color='gray', label='Analysis Failed / No Face')
    ]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Prediction timeline saved to {output_path}")