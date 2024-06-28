import os
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb
plt.style.use("ggplot")


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def get_MNIST_dataset(save_path, download=False, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])

    return torchvision.datasets.MNIST(
        root=save_path,
        train=train,
        transform=transform,
        download=download,
    )


def log_generated_samples(fake_samples, epoch, wandb_object):
    "Generates a 5 x 4 plot of the fake samples"
    fig, axs = plt.subplots(5, 4, figsize=(12, 12))

    for i in range(5):
        for j in range(4):
            index = i * 4 + j
            if index < fake_samples.shape[2]:  # Check if within batch size
                axs[i, j].imshow(fake_samples[:, :, index], cmap='gray')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')  # Turn off empty subplots

    wandb_object.log({f"GAN: generated_samples epoch": wandb.Image(fig)}, step=epoch)


