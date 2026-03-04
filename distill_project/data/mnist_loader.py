# data/mnist_loader.py

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .transforms import get_mnist_transforms


def get_mnist_dataloader(batch_size=128, train=True):

    transform = get_mnist_transforms()

    dataset = MNIST(
        root="./datasets",
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return dataloader