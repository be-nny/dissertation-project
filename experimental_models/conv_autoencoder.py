import os
import matplotlib
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchsummary import summary


matplotlib.use('TkAgg')


class _ConvAutoEncoderModel(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        encoder_layers = []
        for i in range(0, len(layer_sizes) - 1):
            encoder_layers.append(nn.Conv1d(layer_sizes[i], layer_sizes[i+1], kernel_size=2, stride=1, padding=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool1d(kernel_size=2, stride=1, padding=0))

        # defining decoder layers
        layer_sizes = layer_sizes[::-1]
        decoder_layers = []
        for i in range(0, len(layer_sizes) - 1):
            decoder_layers.append(nn.ConvTranspose1d(layer_sizes[i], layer_sizes[i + 1], kernel_size=3, stride=1, padding=1, output_padding=0))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvAutoencoder(_ConvAutoEncoderModel):
    def __init__(self, layer_sizes, uuid, logger, loader, epochs, figures_path):
        super().__init__(layer_sizes)

        self.uuid = uuid
        self.logger = logger
        self.loader = loader
        self.epochs = epochs
        self.figures_path = figures_path

        self.lr = 1e-3
        self.optimiser = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=1000, gamma=0.1)

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            self.logger.info("No GPU available. Training will run on CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_autoencoder(self):
        self.train()

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
                reconstructed = self(x_data)

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
        plt.title("Stacked Autoencoder Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)
        plt.close()

        self.logger.info(f"Saved plot '{path}'")


