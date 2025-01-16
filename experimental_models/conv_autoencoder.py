import math
import os
import matplotlib
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

matplotlib.use('TkAgg')


class _ConvAutoEncoderModel(nn.Module):
    def __init__(self, n_layers, input_shape, latent_dim=10):
        super().__init__()
        self.n_layers = n_layers
        k_s = 3
        s = 2
        p = 1

        # conv encoder
        encoder_layers = []
        self.current_length = input_shape[1]
        for i in range(0, len(n_layers) - 1):
            # work out the size after conv and max pool is applied
            self.current_length = (self.current_length + 2 * p - k_s) // s + 1
            self.current_length = (self.current_length + 2 * p - k_s) // s + 1

            encoder_layers.append(nn.Conv1d(n_layers[i], n_layers[i + 1], kernel_size=k_s, stride=s, padding=p))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.MaxPool1d(kernel_size=k_s, stride=s, padding=p))

        # adding FCN
        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(self.current_length*n_layers[-1], latent_dim))
        encoder_layers.append(nn.LeakyReLU())

        # conv decoder
        reversed_layers = n_layers[::-1]
        decoder_layers = [
            nn.Linear(latent_dim, self.current_length*n_layers[-1]),
            nn.LeakyReLU()
        ]
        self.linear_decoder = nn.Sequential(*decoder_layers)

        decoder_layers = []
        for i in range(0, len(reversed_layers) - 1):
            decoder_layers.append(nn.ConvTranspose1d(reversed_layers[i], reversed_layers[i + 1], kernel_size=4, stride=4, padding=2, output_padding=3))
            decoder_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        linear = self.linear_decoder(encoded)
        linear_reshaped = linear.view(linear.size(0), self.n_layers[-1], self.current_length)
        decoded = self.decoder(linear_reshaped)
        return encoded, decoded


class ConvAutoencoder(_ConvAutoEncoderModel):
    def __init__(self, n_layers, input_shape, uuid, logger, loader, epochs, figures_path):
        super().__init__(n_layers=n_layers, input_shape=input_shape)

        self.uuid = uuid
        self.logger = logger
        self.loader = loader
        self.epochs = epochs
        self.figures_path = figures_path

        self.lr = 1e-3
        self.optimiser = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=1000, gamma=0.1)

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            self.logger.info("No GPU available. Training will run on CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train_autoencoder(self):
        self.train()
        self.to(self.device)

        recon_loss_fn = nn.MSELoss()
        total_losses = []
        best_loss = (np.inf, 0)

        tqdm_loop = tqdm(range(self.epochs), desc="Training Convolutional Autoencoder", unit="epoch")
        for epoch in tqdm_loop:
            losses = []
            for x_data, y_labels in self.loader:
                self.zero_grad()

                # padding the tensor if needed
                pad_size = self.loader.batch_size - x_data.shape[0]
                x_data = nn.functional.pad(x_data, (0, 0, 0, 0, 0, pad_size))

                x_data = x_data.to(self.device)
                latent, reconstructed = self(x_data)

                # the reconstructed data must be the same size, pad it
                reconstructed = nn.functional.pad(reconstructed, (0, 1))
                reconstruction_loss = recon_loss_fn(reconstructed, x_data)

                losses.append(reconstruction_loss.item())
                reconstruction_loss.backward()

                self.optimiser.step()

            mean_loss = np.mean(losses)
            if mean_loss < best_loss[0]:
                best_loss = (mean_loss, epoch)

            tqdm_loop.set_description(f"Training Convolutional Autoencoder - Loss={mean_loss}")
            total_losses.append(mean_loss)

        self.logger.info(f"Training Convolutional Autoencoder complete! Best Loss={best_loss[0]}")

        # plot pre-training loss
        self._plot_loss(epochs=[i for i in range(1, len(total_losses) + 1)], loss_data=total_losses, best_loss=best_loss)

    def _plot_loss(self, loss_data, epochs, best_loss):
        path = os.path.join(self.figures_path, f"{self.uuid}_loss.pdf")

        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss_data, color="blue", label="Loss")
        plt.plot(best_loss[1], best_loss[0], "o", color="red", label=f"Best Loss:{best_loss[0]}")
        plt.title("Convolutional Autoencoder Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)
        plt.close()

        self.logger.info(f"Saved plot '{path}'")


