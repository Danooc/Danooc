#!/usr/bin/env python3
"""
Evaluate a saved DanoocNet checkpoint on the CIFAR-10 test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint best.pt
"""
import argparse

import torch

from danooc.config import Config
from danooc.dataset import get_dataloaders
from danooc.model import DanoocNet
from danooc.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DanoocNet")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = DanoocNet(num_classes=config.num_classes, dropout=0.0)
    trainer = Trainer(model, config, device)
    trainer.load_checkpoint(args.checkpoint)

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
