# main.py

import argparse
import logging
import os
import sys

# # This is crucial for making imports work from the project root
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, PROJECT_ROOT)

from src.engine import run_training, run_evaluation, run_prediction  # noqa: E402
from src.preprocess_data import run_preprocessing  # noqa: E402
from src import config  # noqa: E402

def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Sets up the command-line interface and executes the chosen command."""
    parser = argparse.ArgumentParser(
        description="A Deepfake Detection Algorithm Based on Fourier Transform of Biological Signal (rPPG).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Scan video directories and convert new videos into MVHM images.")
    parser_preprocess.set_defaults(func=lambda args: run_preprocessing())

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help="Train the deepfake detection model on the preprocessed MVHM dataset.")
    parser_train.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if it exists.")
    parser_train.set_defaults(func=lambda args: run_training(resume=args.resume))

    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help="Evaluate the model's performance on the entire preprocessed dataset.")
    parser_eval.add_argument('--model-type', type=str, default='best', choices=['best', 'latest'],
                             help="Specify which model to evaluate: 'best' (highest val_acc) or 'latest' (from last epoch).")
    parser_eval.set_defaults(func=lambda args: run_evaluation(model_type=args.model_type))

    # --- Predict Command ---
    parser_predict = subparsers.add_parser('predict', help="Run a full analysis on a single video file to detect manipulation.")
    parser_predict.add_argument('video_path', type=str, help="Path to the video file for analysis.")
    parser_predict.add_argument('--model-type', type=str, default='best', choices=['best', 'latest'],
                                help="Specify which model to use for prediction: 'best' or 'latest'.")
    parser_predict.set_defaults(func=lambda args: run_prediction(video_path=args.video_path, model_type=args.model_type))

    args = parser.parse_args()

    # Setup logging after parsing args
    setup_logging()

    # Execute the function associated with the chosen command
    args.func(args)

if __name__ == '__main__':
    main()