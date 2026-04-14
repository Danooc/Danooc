"""
Training and evaluation loop encapsulated in the ``Trainer`` class.
"""
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from danooc.model import DanoocNet
from danooc.config import Config


class Trainer:
    """
    Handles model training, evaluation, and checkpoint management.

    Parameters
    ----------
    model : DanoocNet
        The neural network to train.
    config : Config
        Hyperparameters and paths.
    device : torch.device | None
        Accelerator to use; auto-detected when *None*.
    """

    def __init__(
        self,
        model: DanoocNet,
        config: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self, train_loader: DataLoader, test_loader: DataLoader,
    ) -> dict:
        """
        Run the full training loop for ``config.epochs`` epochs.

        Returns a dict with training history (losses and accuracies).
        """
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_acc = 0.0

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch:>3}/{self.config.epochs} "
                f"| train loss {train_loss:.4f} acc {train_acc:.2f}% "
                f"| test loss {test_loss:.4f} acc {test_acc:.2f}% "
                f"| {elapsed:.1f}s"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                self.save_checkpoint("best.pt")

            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

        self.save_checkpoint("last.pt")
        print(f"\nTraining complete — best test accuracy: {best_acc:.2f}%")
        return history

    def _train_one_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """Return ``(loss, accuracy%)`` on the given data loader."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str) -> str:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Loaded checkpoint: {path}")
