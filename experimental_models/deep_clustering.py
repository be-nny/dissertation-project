import os

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from plot_lib import plotter
matplotlib.use('TkAgg')

def _target_distribution(assignments):
    """
    This sharpens the predicted probabilities to focuses more confident assignments
    :param assignments: soft assignments
    :return: target distribution (num_samples, num_clusters)
    """
    targets = assignments.pow(2) / (assignments.sum(dim=0, keepdim=True) + 1e-6)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return targets

class _DEC(nn.Module):
    def __init__(self, n_clusters, latent_dims, ae):
        super().__init__()

        self.n_clusters = n_clusters
        self.latent_dims = latent_dims
        self.ae = ae
        self.alpha = 1

        # create a clustering layer of size (n_clusters, latent_dims)
        # initialise it with a normal distribution
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dims))
        nn.init.xavier_uniform_(self.clustering_layer.data)

    def forward(self, x):
        latent_space, reconstructed = self.ae(x)

        q = self._t_distribution(latent_space)

        return q, latent_space, reconstructed

    def _t_distribution(self, latent_space):
        """
        Use Student's t-Distribution to create a vector of probabilities of this point being assigned to a particular
        cluster. This vector is then normalised.
        :param latent_space: latent_space
        :return: soft assignments (num_samples, num_clusters)
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(latent_space.unsqueeze(1) - self.clustering_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

class DeepClustering:
    def __init__(self, latent_dims, n_clusters, convolutional_ae, loader, logger, figures_path, uuid):
        super().__init__()

        self.logger = logger
        self.figures_path = figures_path
        self.uuid = uuid
        self.loader = loader
        self.latent_dims = latent_dims
        self.n_clusters = n_clusters

        self.convolutional_ae = convolutional_ae
        self.dec = _DEC(n_clusters=self.n_clusters, latent_dims=self.latent_dims, ae=self.convolutional_ae)

        self.lr = 1e-3
        self.optimiser = torch.optim.AdamW(self.dec.parameters(), self.lr, weight_decay=1e-2)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=10, verbose=True)

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            self.logger.info("No GPU available. Training will run on CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs):
        self.dec.train()

        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        recon_loss_fn = nn.MSELoss()

        total_losses = []
        best_loss = np.inf
        tqdm_loop = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in tqdm_loop:
            losses = []

            for x_data, y_true in self.loader:
                self.optimiser.zero_grad()
                self.convolutional_ae.optimiser.zero_grad()

                x_data = x_data.to(self.device)
                q, latent_space, reconstructed = self.dec(x_data)
                p = _target_distribution(q)

                kl_loss = kl_loss_fn(torch.log(q), p.detach())

                # resizing if necessary
                if reconstructed.shape[-1] != x_data.shape[-1]:
                    reconstructed = nn.functional.interpolate(reconstructed, size=(x_data.shape[-1],), mode="nearest")

                recon_loss = recon_loss_fn(reconstructed, x_data)

                loss = recon_loss + kl_loss
                losses.append(loss.item())
                loss.backward()

                self.optimiser.step()
                self.convolutional_ae.optimiser.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss

            total_losses.append(mean_loss)
            tqdm_loop.set_description(f"Training - Loss={mean_loss}")

            self.scheduler.step(mean_loss)

        self._plot_loss(epochs=[i for i in range(1, len(total_losses) + 1)], loss_data=total_losses, best_loss=best_loss)

    def _plot_loss(self, loss_data, epochs, best_loss):
        path = os.path.join(self.figures_path, f"{self.uuid}_dec_loss.pdf")

        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss_data, color="blue", label="Loss")
        plt.plot(best_loss[1], best_loss[0], "o", color="red", label=f"Best Loss:{best_loss[0]}")
        plt.title("Convolutional Autoencoder Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # plt.savefig(path)
        # plt.close()

        self.logger.info(f"Saved plot '{path}'")
