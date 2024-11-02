import torch
import torch.nn as nn


def create_noise(batch_size, latent_space, type="gaussian"):
    if type == "gaussian":
        noise = torch.randn(batch_size, latent_space)
        pass
    elif type == "uniform":
        # We center it around 0 (which is often assumed for data in NN)
        # Furthermore we train the Generator with a tanh (-1, 1) range
        # therefore the noise is in the same range and spans the whole input range
        noise = torch.rand(batch_size, latent_space) * 2 - 1

    else:
        raise ValueError("Please indicate a proper sampling distribution" \
            "Options: gaussian, uniform!")

    return noise


class Generator(nn.Module):

    def __init__(self, latent_dim, img_dim, relu_slope=0.2):
        super().__init__()
        self.img_dim = img_dim
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(relu_slope),
            nn.Linear(256, 512),
            nn.LeakyReLU(relu_slope),
            nn.Linear(512, img_dim * img_dim),
            # Remember. MNIST is in range (-1, 1), so Tanh gives exactly this
            nn.Tanh(),
        )

    def forward(self, noise):
        fake_x = self.gen(noise)
        return fake_x.reshape(-1, self.img_dim, self.img_dim)


class Discriminator(nn.Module):

    def __init__(self, img_dim, relu_slope=0.2):
        super().__init__()
        self.img_dim = img_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.img_dim * self.img_dim, 512),
            nn.LeakyReLU(relu_slope),
            nn.Linear(512, 256),
            nn.LeakyReLU(relu_slope),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.discriminator(x.reshape(-1, self.img_dim * self.img_dim))
        return logits