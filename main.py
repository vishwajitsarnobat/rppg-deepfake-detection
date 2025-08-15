# main.py

import argparse
import logging
import os
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
    download_dlib_model()

    parser = argparse.ArgumentParser(
        description="A Deepfake Detection System based on rPPG Biological Signals (Paper Implementation).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    parser_preprocess = subparsers.add_parser('preprocess', help="Segment all videos and convert them into MVHM images.")
    parser_preprocess.set_defaults(func=lambda args: run_preprocessing())

    parser_train = subparsers.add_parser('train', help="Train the deepfake detector model.")
    parser_train.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint.")
    parser_train.set_defaults(func=lambda args: run_training(resume=args.resume))

    parser_eval = subparsers.add_parser('evaluate', help="Evaluate the best model on the validation set.")
    parser_eval.set_defaults(func=lambda args: run_evaluation())

    parser_predict = subparsers.add_parser('predict', help="Run a full, segmented analysis on a single video file.")
    parser_predict.add_argument('video_path', type=str, help="Path to the video file to analyze.")
    parser_predict.set_defaults(func=lambda args: run_prediction(video_path=args.video_path))

    args = parser.parse_args()
    setup_logging()
    args.func(args)

if __name__ == '__main__':
    main()