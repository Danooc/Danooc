#!/usr/bin/env python3
"""
Classify a single image using a trained DanoocNet checkpoint.

Usage:
    python scripts/predict.py path/to/image.png
    python scripts/predict.py photo.jpg --checkpoint last.pt
"""
import argparse

from danooc.config import Config
from danooc.predict import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an image with DanoocNet")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    config = Config(checkpoint_dir=args.checkpoint_dir)
    class_name, confidence = predict_image(args.image, args.checkpoint, config)
    print(f"Prediction: {class_name} ({confidence:.1f}% confidence)")


if __name__ == "__main__":
    main()
