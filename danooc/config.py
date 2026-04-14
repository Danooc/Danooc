"""
Hyperparameters and project-wide configuration.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- data ---
    data_dir: str = "./data"
    num_workers: int = 2
    batch_size: int = 64

    # --- model ---
    num_classes: int = 10
    dropout: float = 0.25

    # --- training ---
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 7
    lr_gamma: float = 0.1

    # --- checkpoints ---
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5

    # --- CIFAR-10 class names ---
    class_names: List[str] = field(default_factory=lambda: [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ])
