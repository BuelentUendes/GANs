import torch
import torch.nn as nn


def create_noise(batch_size, latent_space, type="gaussian"):
    """
    Create noise tensor for input to the generator of the GAN.

    Args:
        batch_size (int): Number of samples in the batch.
        latent_space (int): Dimension of the latent space.
        type (str, optional): Type of noise distribution. Defaults to "gaussian".

    Returns:
        torch.Tensor: Noise tensor of shape (batch_size, latent_space).

    Raises:
        ValueError: If an invalid noise type is specified.
    """
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
    """
    Generator network for GAN.

    Args:
        latent_dim (int): Dimension of the latent space.
        img_dim (int): Dimension of the output image (assumed to be square).
        relu_slope (float, optional): Negative slope of the LeakyReLU activation. Defaults to 0.2.
    """

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
        """
        Forward pass of the generator.

        Args:
            noise (torch.Tensor): Input noise tensor.

        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, img_dim, img_dim).
        """
        fake_x = self.gen(noise)
        return fake_x.reshape(-1, self.img_dim, self.img_dim)


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.

    Args:
        img_dim (int): Dimension of the input image (assumed to be square).
        relu_slope (float, optional): Negative slope of the LeakyReLU activation. Defaults to 0.2.
    """

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
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Probability of the input being real (1) or fake (0).
        """
        logits = self.discriminator(x.reshape(-1, self.img_dim * self.img_dim))
        return logits