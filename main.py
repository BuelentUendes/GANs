# Standard library imports
import argparse
import yaml

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Local application imports
from src.vanilla_GAN import create_noise, Generator, Discriminator
from utils.helper_functions import create_directory, get_MNIST_dataset, log_generated_samples
from utils.helper_path import MNIST_PATH

IMG_DIM = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_optimizers(discriminator: nn.Module, generator: nn.Module, lr: float) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """
    Creates optimizers for the discriminator and generator networks.

    Args:
        discriminator: The discriminator neural network
        generator: The generator neural network
        lr (float): Learning rate for both optimizers

    Returns:
        tuple: (discriminator_optimizer, generator_optimizer)
    """
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr)
    return d_optimizer, g_optimizer


def train_discriminator(
    x: torch.Tensor,
    discriminator: nn.Module,
    generator: nn.Module,
    d_optimizer: torch.optim.Adam,
    latent_space_dim: int,
    device: str = DEVICE
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Performs one training step for the discriminator.

    Args:
        x (torch.Tensor): Batch of real images
        discriminator: The discriminator neural network
        generator: The generator neural network
        d_optimizer: Optimizer for the discriminator
        latent_space_dim (int): Dimension of the latent space for noise generation
        device (str): Device to run computations on ('cuda' or 'cpu')

    Returns:
        tuple: (discriminator_loss, probabilities_real, probabilities_fake)
    """
    loss_fn = nn.BCELoss()
    discriminator.zero_grad()
    batch_size = x.size(0)
    # Sample noise
    noise = create_noise(batch_size, latent_space_dim).to(device)
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


def train_generator(
    x: torch.Tensor,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Adam,
    latent_space_dim: int,
    device: str = DEVICE
) -> float:
    """
    Performs one training step for the generator.

    Args:
        x (torch.Tensor): Batch of real images (used only for batch size)
        generator: The generator neural network
        discriminator: The discriminator neural network
        g_optimizer: Optimizer for the generator
        latent_space_dim (int): Dimension of the latent space for noise generation
        device (str): Device to run computations on ('cuda' or 'cpu')

    Returns:
        float: Generator loss value
    """
    loss_fn = nn.BCELoss()
    generator.zero_grad()
    batch_size = x.size(0)
    noise = create_noise(batch_size, latent_space_dim).to(device)
    samples_fake = generator(noise)
    g_labels_real = torch.ones(batch_size, 1, device=device)

    d_proba_fake = discriminator(samples_fake)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    g_loss.backward()
    g_optimizer.step()

    return g_loss.cpu().item()


def create_samples(generator: nn.Module, noise: torch.Tensor) -> torch.Tensor:
    """
    Generates image samples using the generator.

    Args:
        generator: The generator neural network
        noise (torch.Tensor): Input noise tensor

    Returns:
        torch.Tensor: Generated images normalized to range [0, 1]
    """
    generated_sample = generator(noise).detach().cpu()
    # We need to reshape the data in the HWC as numpy expects it and denormalize it in the range 0, 1
    images = generated_sample.permute(1, 2, 0)

    return (images + 1) / 2


def main(args: argparse.Namespace) -> None:
    """
    Main training loop for the GAN.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - wandb_logging (bool): Whether to log to Weights & Biases
            - batch_size (int): Batch size for training
            - latent_space_dim (int): Dimension of the latent space
            - lr (float): Learning rate
            - epochs (int): Number of training epochs
            - visualized_img (int): Number of images to visualize
            - verbose (bool): Whether to print progress
            - sweep (bool): Whether this is part of a hyperparameter sweep
    """
    if args.wandb_logging:
        wandb.init(
            project='GAN_plaground',
            entity='b-uendes',
            name='GAN_playground_' + wandb.util.generate_id(),
            reinit=True,
            notes='Visualization of GAN runs'
        )

    create_directory(MNIST_PATH)
    mnist_dataset = get_MNIST_dataset(MNIST_PATH, download=True)

    # Makes it easier to go in batches to work with dataloader
    mnist_dataloader = DataLoader(mnist_dataset, args.batch_size, shuffle=True)
    # We will need a validation set in case we sweep

    generator = Generator(args.latent_space_dim, IMG_DIM).to(DEVICE)
    discriminator = Discriminator(IMG_DIM).to(DEVICE)
    d_optimizer, g_optimizer = get_optimizers(discriminator, generator, lr=args.lr)

    # Sample fix noise, so I can see how the images evolve over time for a fixed noise sample
    fixed_noise = create_noise(args.visualized_img, args.latent_space_dim).to(DEVICE)

    all_epoch_proba_real = []
    all_epoch_proba_fake = []
    all_epoch_generator_loss = []
    epoch_discriminator_loss = []

    for epoch in range(1, args.epochs + 1):
        if args.verbose:
            print(f"Train GAN model: {epoch}/{args.epochs}")
        discriminator_loss, generator_loss = [], []
        probability_real, probability_fake = [], []

        for i, (x, _) in enumerate(mnist_dataloader):
            total_loss_d, d_proba_real, d_proba_fake = \
                train_discriminator(x, discriminator, generator, d_optimizer, args.latent_space_dim,
                                    device=DEVICE)
            total_loss_g = train_generator(x, generator, discriminator, g_optimizer, args.latent_space_dim,
                                           device=DEVICE)

            discriminator_loss.append(total_loss_d)
            generator_loss.append(total_loss_g)
            probability_real.append(d_proba_real.cpu().mean())
            probability_fake.append(d_proba_fake.cpu().mean())

        epoch_loss_generator = torch.tensor(generator_loss).mean()
        epoch_loss_discriminator = torch.tensor(discriminator_loss).mean()
        epoch_probability_fake = torch.tensor(probability_fake).mean()
        epoch_probability_real = torch.tensor(probability_real).mean()

        if args.verbose:
            print(f"\nloss_generator: {epoch_loss_generator} \nloss discriminator: {epoch_loss_discriminator}"
                  f"\nprobability_fake: {epoch_probability_fake} \nprobability_real: {epoch_probability_real}")

        all_epoch_proba_real.append(epoch_probability_real)
        all_epoch_proba_fake.append(epoch_probability_fake)
        all_epoch_generator_loss.append(epoch_loss_generator)
        epoch_discriminator_loss.append(epoch_loss_discriminator)

        # Plot if wandb object is provided:
        if args.wandb_logging:
            fake_samples = create_samples(generator, fixed_noise)
            log_generated_samples(fake_samples, epoch=epoch, wandb_object=wandb)
            wandb.log(
                {
                    "loss_generator": epoch_loss_generator,
                    "loss_discriminator": epoch_loss_discriminator,
                    "probability_fake": epoch_probability_fake,
                    "probability_real": epoch_probability_real,
                }
            )

    if args.sweep:
        loss_fn = nn.BCELoss()
        fixed_noise_val = create_noise(100, args.latent_space_dim).to(DEVICE)
        samples_fake_val = generator(fixed_noise_val)
        g_labels_real = torch.ones(100, 1, device=DEVICE)

        d_proba_fake = discriminator(samples_fake_val)

        g_loss_val = loss_fn(d_proba_fake, g_labels_real)
        wandb.log(
            {
                "val_loss_discriminator": g_loss_val
            }
        )

    if args.wandb_logging:
        wandb.finish()


def sweep_main() -> None:
    """
    Entry point for hyperparameter sweeping using Weights & Biases.
    Initializes a wandb run and calls main() with sweep configuration.
    """
    wandb.init()
    args = argparse.Namespace(
        epochs=2,
        batch_size=64,
        latent_space_dim=wandb.config.latent_space_dim,
        lr=wandb.config.lr,
        seed=7,
        wandb_logging=False,
        visualized_img=20,
        sweep=True,
        verbose=True
    )
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        help='Override default config setting for iterations we want to train the algorithm with',
                        default=1, type=int)
    parser.add_argument('--batch_size', help="Override default config batch size to train the algorithm with",
                        default=64, type=int)
    parser.add_argument("--latent_space_dim", help="latent space dimension for noise", default=20, type=int)
    parser.add_argument("--lr", help="learning rate", default=3e-4, type=float)
    parser.add_argument("--seed", help='seed number', default=7, type=int)
    parser.add_argument("--wandb_logging", help='Boolean, if TRUE then wandb logging will be enabled',
                        action='store_true')
    parser.add_argument("--visualized_img", help="Number of visualized images", type=int, default=20)
    parser.add_argument("--sweep", help="Boolean, indicating if hyperparameter sweep via W&B is used",
                        action="store_true")
    parser.add_argument("--number_sweeps", type=int, help="Number of sweeps to do", default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Seed setting for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.sweep:
        # Set up your default hyperparameters
        with open("./config/sweep.yaml") as file:
            sweep_config = yaml.load(file, Loader=yaml.FullLoader)

        sweep_id = wandb.sweep(sweep=sweep_config)
        wandb.agent(sweep_id, function=sweep_main, count=args.number_sweeps)

    else:
        main(args)





