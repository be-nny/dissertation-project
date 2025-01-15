import os
import matplotlib
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

matplotlib.use('TkAgg')

class StackedAutoEncoderModel(nn.Module):
    def __init__(self, hidden_layers, dropout_rate):
        super().__init__()

        # encoder layers
        encoder_layers = []
        for i in range(0, len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))

        # decoder layers
        decoder_layers = []
        hidden_layers = hidden_layers[::-1]
        for i in range(0, len(hidden_layers) - 1):
            decoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            decoder_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoder(StackedAutoEncoderModel):
    def __init__(self, uuid, hidden_layers, logger, loader, epochs, figures_path, dropout_rate: float = 0.3):
        super().__init__(hidden_layers, dropout_rate)
        self.uuid = uuid
        self.epochs = epochs
        self.logger = logger
        self.loader = loader
        self.figures_path = figures_path

        self.input_dim = hidden_layers[0]
        self.latent_dim = hidden_layers[-1]

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

        recon_loss_fn = nn.MSELoss()
        total_losses = []
        best_loss = (np.inf, 0)

        tqdm_loop = tqdm(range(self.epochs), desc="Training Autoencoder", unit="epoch")
        for epoch in tqdm_loop:
            losses = []
            for x_data, y_labels in self.loader:
                self.zero_grad()
                x_data = x_data.to(self.device)
                latent, reconstructed = self(x_data)

                reconstruction_loss = recon_loss_fn(reconstructed, x_data)

                losses.append(reconstruction_loss.item())
                reconstruction_loss.backward()

                self.optimiser.step()

            mean_loss = np.mean(losses)
            if mean_loss < best_loss[0]:
                best_loss = (mean_loss, epoch)

            tqdm_loop.set_description(f"Training Autoencoder - Loss={mean_loss}")
            total_losses.append(mean_loss)
            # self.scheduler.step()

        self.logger.info(f"Training Autoencoder complete! Best Loss={best_loss[0]}")

        # plot pre-training loss
        self._plot_loss(epochs=[i for i in range(1, len(total_losses) + 1)], loss_data=total_losses, best_loss=best_loss)

    def regularisation_term(self, latent_space, lamda_term: float = 1e-3):
        """
        This is term used for contractive loss that penalises the encoding layer for being
        too sensitive to small changes in the input data.

        λ * ||J_f(x)||²

        - λ is a hyperparam that controls the strength at which the encoder is penalised
        - J_f(x) is the Jacobian matrix of the latent_Space

        :param latent_space: latent representation of the encoded data
        :param lamda_term: hyperparam that controls the strength at which the encoder is penalised
        :return:
        """

        jacobian = torch.zeros(self.loader.batch_size, self.latent_dim, self.input_dim, device=self.device)
        for i in range(self.latent_dim):
            grad = torch.autograd.grad(
                outputs=latent_space[:, i],
                inputs=latent_space,
                grad_outputs=torch.ones_like(latent_space[:, i]),
                create_graph=True,
                retain_graph=True,
            )[0]
            jacobian[:, i, :] = grad.view(self.input_dim, -1)

        jacobian_norm = torch.sum(jacobian**2)

        return lamda_term * jacobian_norm

    def model(self):
        return self

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

    def _plot_latent_space(self):
        latent_space = []
        labels = []

        for x, y in self.loader:
            x = x.to(self.device)
            latent, _ = self(x)
            latent_space.extend(latent.detach().cpu().numpy())
            labels.extend(y.cpu().numpy())

        latent_space = np.array(latent_space)
        labels = np.array(labels)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            latent_space[:, 0], latent_space[:, 1], latent_space[:, 2],
            c=labels, cmap='viridis', alpha=0.7, s=10
        )

        cbar = plt.colorbar(scatter, ax=ax, label="Cluster Labels")
        ax.set_title("3D Plot of Latent Space")
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.set_zlabel("Axis 3")
        path = os.path.join(self.figures_path, "latent_space.pdf")
        plt.savefig(path)
        self.logger.info(f"Saved plot '{path}'")
        plt.show()
