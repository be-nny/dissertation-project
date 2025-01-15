import logging
import os
import matplotlib
import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from . import utils
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from . import stacked_autoencoder
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

class DEC(nn.Module):
    def __init__(self, lstm_ae, latent_space_dims: int, n_clusters: int = 10):
        super().__init__()

        self.lstm_ae = lstm_ae
        self.alpha = 0.001

        # create a clustering layer
        # this is the learnable parameter for the cluster centroids
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_space_dims))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)

    def forward(self, x):
        latent_space, reconstructed = self.lstm_ae(x)
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

class ClusteringModel:
    def __init__(self, dataset_loader: utils.Loader, logger: logging.Logger, n_clusters: int = 10, pre_train_epochs: int = 500, latent_dim: int = 3):
        self.dataset_loader = dataset_loader
        self.figures_path = self.dataset_loader.get_figures_path()
        self.dataloader = self.dataset_loader.load(split_type="train", batch_size=128)
        self.logger = logger
        self.n_clusters = n_clusters

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            self.logger.info("No GPU available. Training will run on CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # lstm autoencoder
        input_shape = self.dataset_loader.get_input_shape()
        self.lstm_ae = autoencoder.LSTMAutoencoder(input_dim=input_shape[1], hidden_dim=128, latent_dim=latent_dim, num_layers=4).to(self.device)
        self.lstm_lr = 1e-4
        self.lstm_optimiser = torch.optim.AdamW(self.lstm_ae.parameters(), self.lstm_lr, weight_decay=1e-3)
        self.lstm_scheduler = lr_scheduler.StepLR(self.lstm_optimiser, step_size=100, gamma=0.1)
        self._pre_train(n_epochs=pre_train_epochs)

        # dec
        self.dec = DEC(self.lstm_ae, latent_dim, n_clusters)
        self.dec_lr = 1e-3
        self.dec_optimiser = torch.optim.AdamW(self.dec.parameters(), lr=self.dec_lr)
        self.dec_scheduler = lr_scheduler.StepLR(self.dec_optimiser, step_size=100, gamma=0.1)

    def _pre_train(self, n_epochs: int = 500):
        self.lstm_ae.train()

        recon_loss_fn = nn.MSELoss()
        total_losses = []
        best_loss = np.inf

        tqdm_loop = tqdm(range(n_epochs), desc="Pre-training Autoencoder", unit="epoch")
        for _ in tqdm_loop:
            losses = []
            for x_data, y_labels in self.dataloader:
                self.lstm_ae.zero_grad()
                x_data = x_data.to(self.device)
                latent, reconstructed = self.lstm_ae(x_data)

                reconstruction_loss = recon_loss_fn(reconstructed, x_data)
                losses.append(reconstruction_loss.item())
                reconstruction_loss.backward()

                self.lstm_optimiser.step()

            mean_loss = np.mean(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss

            tqdm_loop.set_description(f"Pre-training Autoencoder - Loss={mean_loss}")
            total_losses.append(mean_loss)

            self.lstm_scheduler.step()

        self.logger.info(f"Pre-training Autoencoder complete! Best Loss={best_loss}")

        # plot pre-training loss
        self._plot_figure(title="Pre-training Loss", path=os.path.join(self.figures_path, "pre-training_loss.pdf"), epochs=[i for i in range(1, len(total_losses) + 1)], loss=total_losses)
        self._plot_latent_space()
        return self

    def _plot_latent_space(self):
        latent_space = []
        labels = []

        for x, y in self.dataloader:
            x = x.to(self.device)
            latent, _ = self.lstm_ae(x)
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


    def train(self, clustering_epochs: int = 500, update_freq: int = 5):
        self.dec.train()

        # set the loss functions to use
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        recon_loss_fn = nn.MSELoss()

        total_losses = []
        total_nmi = []
        total_ari = []

        # initialise the centroids
        total_data = []
        total_labels = []
        for x, y in self.dataloader:
            total_data.extend(x)
            total_labels.extend(y)
        total_data = torch.tensor(np.array(total_data), dtype=torch.float32).to(self.device)

        kmeans = KMeans(n_clusters=self.n_clusters)
        latent_space, _ = self.lstm_ae(total_data)
        y_pred = kmeans.fit_predict(latent_space.detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        self.dec.clustering_layer.data = torch.tensor(cluster_centers, dtype=torch.float32).to(self.device)
        y_prev = y_pred

        best_loss = np.inf
        tqdm_loop = tqdm(range(clustering_epochs), desc="Training", unit="epoch")
        for epoch in tqdm_loop:
            losses = []

            # updating cluster centers
            if epoch % update_freq == 0 and (epoch > 15 or epoch == 0):
                q, _, _ = self.dec(total_data)
                p = self._target_distribution(q)

                # pick the largest probability from the t distribution, this will be the predicted label
                # delta_label is the fraction of labels that have switched assignments - lower means more convergence
                y_pred = q.detach().cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_prev).astype(np.float32) / y_prev.shape[0]

                nmi = sklearn.metrics.normalized_mutual_info_score(total_labels, y_pred)
                ari = sklearn.metrics.adjusted_rand_score(total_labels, y_pred)
                y_prev = y_pred

                total_nmi.append([nmi, epoch])
                total_ari.append([ari, epoch])

            st_ptn = 0
            end_ptn = self.dataloader.batch_size
            for x_data, y_true in self.dataloader:
                self.dec_optimiser.zero_grad()
                x_data = x_data.to(self.device)
                q, latent_space, reconstructed = self.dec(x_data)

                # work out loss
                kl_loss = kl_loss_fn(torch.log(q + 1e-10), torch.tensor(p.detach().cpu().numpy()[st_ptn:end_ptn]).to(self.device))
                reconstruction_loss_1 = recon_loss_fn(reconstructed, x_data)
                loss = kl_loss * 2 + reconstruction_loss_1

                loss.backward()
                losses.append(loss.item())

                # prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.dec.parameters(), max_norm=1)

                # update weights
                self.dec_optimiser.step()

                st_ptn += self.dataloader.batch_size
                end_ptn += self.dataloader.batch_size

            mean_loss = np.mean(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss

            tqdm_loop.set_description(f"Training - Loss={mean_loss}, NMI={nmi}, ARI={ari}, DELTA={delta_label}")
            total_losses.append(mean_loss)
            self.dec_scheduler.step()

        self.logger.info(f"Training complete! Best Loss={best_loss}")

        # plot training loss
        self._plot_figure(title="Training Loss", path=os.path.join(self.figures_path, "training_loss.pdf"), epochs=[i for i in range(1, len(total_losses) + 1)], loss=total_losses)
        self._plot_figure(title="Normalised Mutual Information Scores", path=os.path.join(self.figures_path, "nmi_scores.pdf"), epochs=[val[1] for val in total_nmi], nmi=[val[0] for val in total_nmi])
        self._plot_figure(title="Adjusted Rand Index Scores", path=os.path.join(self.figures_path, "ari_scores.pdf"), epochs=[val[1] for val in total_ari], nmi=[val[0] for val in total_ari])

        return self

    def _plot_figure(self, title, path, **kwargs):
        plt.figure(figsize=(10, 10))

        x_label, x_val = list(kwargs.items())[0]
        y_label, y_val = list(kwargs.items())[1]

        plt.plot(x_val, y_val)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(path)
        plt.close()

        self.logger.info(f"Saved plot '{path}'")

    def _target_distribution(self, assignments):
        """
        This sharpens the predicted probabilities to focuses more confident assignments

        :param assignments: soft assignments
        :return: target distribution (num_samples, num_clusters)
        """

        targets = assignments.pow(2) / (assignments.sum(dim=0, keepdim=True) + 1e-6)
        targets = targets / targets.sum(dim=1, keepdim=True)

        return targets
