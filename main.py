# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
from os.path import abspath
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.helper_path import MNIST_PATH
from utils.helper_functions import create_directory, get_MNIST_dataset
from torch.utils.data import DataLoader
from src.vanilla_GAN import create_noise, Generator, Discriminator
import numpy as np

BATCH_SIZE = 32
LATENT_SPACE_DIM = 20
IMG_DIM = 28
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
loss_fn = nn.BCELoss()

def get_optimizers(discriminator, generator, lr=LR):
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    return d_optimizer, g_optimizer


def train_discriminator(x, discriminator, generator, d_optimizer, device=DEVICE):
    discriminator.zero_grad()
    batch_size = x.size(0)
    # Sample noise
    noise = create_noise(batch_size, LATENT_SPACE_DIM).to(device)
    samples_fake = generator.forward(noise).to(DEVICE)

    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_proba_real = discriminator(x.to(device))
    d_proba_fake = discriminator(samples_fake)

    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    total_loss_d = d_loss_fake + d_loss_real
    total_loss_d.backward()
    d_optimizer.step()

    return total_loss_d.cpu().item(), d_proba_real, d_proba_fake


def train_generator(x, generator, discriminator, g_optimizer, device=DEVICE):
    generator.zero_grad()
    batch_size = x.size(0)
    noise = create_noise(batch_size, LATENT_SPACE_DIM).to(device)
    samples_fake = generator(noise)
    g_labels_real = torch.ones(batch_size, 1, device=device)

    d_proba_fake = discriminator(samples_fake)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    g_loss.backward()
    g_optimizer.step()

    return g_loss.cpu().item()


def create_samples(generator, noise):
    generated_sample = generator(noise)
    # We need to reshape the data in the HWC as numpy expects it and denormalize it in the range 0, 1
    images = generated_sample.permute(1, 2, 0)

    return (images+1) / 2


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)

    create_directory(MNIST_PATH)
    mnist_dataset = get_MNIST_dataset(MNIST_PATH, download=True)

    # Makes it easier to go in batches to work with dataloader
    mnist_dataloader = DataLoader(mnist_dataset, BATCH_SIZE, shuffle=False)

    generator = Generator(LATENT_SPACE_DIM, IMG_DIM).to(DEVICE)
    discriminator = Discriminator(IMG_DIM).to(DEVICE)
    d_optimizer, g_optimizer = get_optimizers(discriminator, generator)

    # Sample fix noise, so I can see how the images evolve over time for a fixed noise sample
    # , here we take only 5 images that is enough
    fixed_noise = create_noise(5, LATENT_SPACE_DIM).to(DEVICE)

    all_epoch_samples = []
    all_epoch_proba_real = []
    all_epoch_proba_fake = []
    all_epoch_generator_loss = []
    epoch_discriminator_loss = []

    for epoch in range(1, EPOCHS+1):
        print(f"Train {epoch}/{EPOCHS}")
        discriminator_loss, generator_loss = [], []
        probability_real, probability_fake = [], []

        for i, (x, _) in enumerate(mnist_dataloader):
            total_loss_d, d_proba_real, d_proba_fake = \
                train_discriminator(x, discriminator, generator, d_optimizer, device=DEVICE)
            total_loss_g = train_generator(x, generator, discriminator, g_optimizer, device=DEVICE)

            discriminator_loss.append(total_loss_d)
            generator_loss.append(total_loss_g)
            probability_real.append(d_proba_real.cpu().mean())
            probability_fake.append(d_proba_fake.cpu().mean())

        epoch_loss_generator = torch.tensor(generator_loss).mean()
        epoch_loss_discriminator = torch.tensor(discriminator_loss).mean()
        epoch_probability_fake = torch.tensor(probability_fake).mean()
        epoch_probability_real = torch.tensor(probability_real).mean()

        print(f"\nloss_generator: {epoch_loss_generator} \nloss discriminator: {epoch_loss_discriminator}"
              f"\nprobability_fake: {epoch_probability_fake} \nprobability_real: {epoch_probability_real}")

        all_epoch_samples.append(create_samples(generator, fixed_noise))
        all_epoch_proba_real.append(epoch_probability_real)
        all_epoch_proba_fake.append(epoch_probability_fake)
        all_epoch_generator_loss.append(epoch_loss_generator)
        epoch_discriminator_loss.append(epoch_loss_discriminator)





