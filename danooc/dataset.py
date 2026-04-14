"""
Dataset utilities — download CIFAR-10 and build DataLoaders with standard
augmentations for training and normalisation for evaluation.
"""
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, test_loader)`` for CIFAR-10.

    Training data is augmented with random horizontal flips and crops;
    test data uses only normalisation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader
