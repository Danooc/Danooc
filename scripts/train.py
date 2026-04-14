#!/usr/bin/env python3
"""
Train DanoocNet on CIFAR-10.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 30 --batch-size 128 --lr 0.0005
"""
import argparse

import torch

from danooc.config import Config
from danooc.dataset import get_dataloaders
from danooc.model import DanoocNet
from danooc.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DanoocNet on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
          f"lr={config.learning_rate}, dropout={config.dropout}")

    train_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    model = DanoocNet(num_classes=config.num_classes, dropout=config.dropout)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
