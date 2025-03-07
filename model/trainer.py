import numpy as np
import torch
import logging

from torch import nn, optim
from tqdm import tqdm
from model import models, utils

class RunningStats:
    """
    Keeps track of the moving average of a loss function. This is used when combining two or more loss functions so that
    their values both contribute equally to the loss function.
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.mean = None
        self.var = None

    def update(self, val):
        if self.mean is None:
            self.mean = val
            self.var = torch.zeros_like(val)
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * val
            self.var = self.momentum * self.var + (1 - self.momentum) * (val-self.mean)**2

    def stats(self):
        std = torch.sqrt(self.var + 1e-8)
        return self.mean, std

def train_autoencoder(epochs: int, autoencoder: models.Conv1DAutoencoder, batch_loader: utils.DataLoader, batch_size: int, logger: logging.Logger, path: str) -> None:
    """
    Trains a convolutional autoencoder, ready for the DEC model.

    :param epochs: number of training epochs
    :param autoencoder: autoencoder model
    :param batch_loader: batch loader
    :param batch_size: batch size
    :param logger: logger
    :param path: path to save model
    """

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        logger.info("No GPU available. Training will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder.train()
    autoencoder.to(device)

    # training parameters
    lr = 1e-3
    optimiser = torch.optim.AdamW(autoencoder.parameters(), lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    loss_vals = []
    best_loss = (np.inf, 0)

    tqdm_loop = tqdm(range(epochs), desc="Training Conv Autoencoder", unit="epoch")
    for epoch in tqdm_loop:
        losses = []
        for x, y in batch_loader:
            autoencoder.zero_grad()

            pad_size = batch_size - x.shape[0]
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad_size))
            x = x.to(device)

            latent_space, reconstructed = autoencoder(x)
            reconstructed = nn.functional.interpolate(reconstructed, size=(x.shape[-1],), mode="nearest")
            loss = loss_fn(reconstructed, x)

            losses.append(loss.item())
            loss.backward()
            optimiser.step()

        mean_loss = np.mean(losses)
        loss_vals.append(mean_loss)

        if mean_loss < best_loss[0]:
            best_loss = (mean_loss, epoch)
        tqdm_loop.set_description(f"Training Convolutional Autoencoder - Loss={mean_loss}")

    logger.info(f"Training complete! Best loss: {best_loss[0]}")
    torch.save(autoencoder.state_dict(), path)
    logger.info(f"Saved weights to '{path}'")


def train_dec(epochs: int, dec: models.DEC, batch_loader: utils.DataLoader, logger: logging.Logger, path: str) -> None:
    """
    Trains a Deep Embedded Clustering implementation using a pre-trained convolutional autoencoder. The loss function
    is a combination of the reconstruction loss of the convolutional autoencoder, and the KL divergence. These values
    are scaled so that they contribute equally to the total loss govenered by the moving average and std.

    :param epochs: number of training epochs
    :param dec: DEC instance
    :param batch_loader: batch loader
    :param logger: logger
    :param path: path to save model
    """

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        logger.info("No GPU available. Training will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec.train()

    # moving mean/std trackers
    recon_stats = RunningStats()
    kl_stats = RunningStats()

    # training parameters
    lr = 1e-3
    optimiser = torch.optim.AdamW(dec.parameters(), lr, weight_decay=1e-2)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    recon_loss_fn = nn.MSELoss()

    loss_vals = []
    best_loss = (np.inf, 0)

    tqdm_loop = tqdm(range(epochs), desc="Training DEC", unit="epoch")

    for epoch in tqdm_loop:
        losses = []
        for x, y in batch_loader:
            optimiser.zero_grad()
            x = x.to(device)
            q, latent_space, reconstructed = dec(x)
            p = models.target_distribution(q)

            if reconstructed.shape[-1] != x.shape[-1]:
                reconstructed = nn.functional.interpolate(reconstructed, size=(x.shape[-1],), mode="nearest")

            kl_loss = kl_loss_fn(torch.log(q), p.detach())
            recon_loss = recon_loss_fn(reconstructed, x)

            # scale both loss values equally with moving average tracker
            recon_stats.update(recon_loss.detach())
            kl_stats.update(kl_loss.detach())
            recon_mean, recon_std = recon_stats.stats()
            kl_mean, kl_std = kl_stats.stats()
            norm_recon = (recon_loss - recon_mean) / (recon_std + 1e-8)
            norm_kl = (kl_loss - kl_mean) / (kl_std + 1e-8)

            loss = norm_recon + norm_kl
            losses.append(loss.item())
            loss.backward()
            optimiser.step()

        mean_loss = np.mean(losses)
        if mean_loss < best_loss[0]:
            best_loss = (mean_loss, epoch)

        loss_vals.append(mean_loss)
        tqdm_loop.set_description(f"Training - Loss={mean_loss}")

    logger.info(f"Training complete! Best loss: {best_loss[0]}")
    torch.save(dec.state_dict(), path)
    logger.info(f"Saved weights to '{path}'")
