import numpy as np
import torch
import logging

from torch import nn
from tqdm import tqdm
from model import models, utils

def train_autoencoder(epochs, autoencoder: models.Conv1DAutoencoder, batch_loader: utils.DataLoader, batch_size: int, logger: logging.Logger, path: str):
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

