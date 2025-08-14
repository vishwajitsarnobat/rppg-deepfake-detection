# main.py

import argparse
import logging
import sys

from src.engine import run_training, run_evaluation, run_prediction
from src.preprocess_data import run_preprocessing
from src import config
from src.utils import download_dlib_model

def setup_logging():
    """Configures logging to a file and the console with a professional format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Sets up and runs the main CLI for the project."""
    # --- NEW: Automatically download dlib model if it's missing ---
    download_dlib_model()

    parser = argparse.ArgumentParser(
        description="A Deepfake Detection System based on rPPG Biological Signals.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Scan videos and convert them into MVHM images.")
    parser_preprocess.set_defaults(func=lambda args: run_preprocessing())

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help="Train the deepfake detector model.")
    parser_train.set_defaults(func=lambda args: run_training())

    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help="Evaluate the model on the preprocessed dataset.")
    parser_eval.add_argument('--model-path', type=str, default=None, help="Path to a specific model file to evaluate.")
    parser_eval.set_defaults(func=lambda args: run_evaluation(model_path=args.model_path))

    # --- Predict Command ---
    parser_predict = subparsers.add_parser('predict', help="Run a full analysis on a single video file.")
    parser_predict.add_argument('video_path', type=str, help="Path to the video file to analyze.")
    parser_predict.add_argument('--model-type', type=str, default='best', choices=['best', 'latest'],
                                help="Which model to use for prediction: 'best' or 'latest'.")
    parser_predict.set_defaults(func=lambda args: run_prediction(video_path=args.video_path, model_type=args.model_type))

    args = parser.parse_args()
    setup_logging()
    args.func(args)

if __name__ == '__main__':
    main()