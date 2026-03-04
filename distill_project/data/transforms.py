# data/transforms.py

import torchvision.transforms as transforms


def get_mnist_transforms():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform