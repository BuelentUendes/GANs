import os
import torchvision
from torchvision import transforms


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def get_MNIST_dataset(save_path, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])

    return torchvision.datasets.MNIST(
        root=save_path,
        train=True,
        transform=transform,
        download=download,
    )

