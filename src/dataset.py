from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_transforms(train: bool = True):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    return transforms.Compose(transform_list)


def get_mnist_datasets(data_dir: str | Path = "data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    data_dir = Path(data_dir)
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_mnist_transforms(train=True),
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_mnist_transforms(train=False),
    )
    return train_dataset, test_dataset


def get_mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_mnist_datasets(data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
