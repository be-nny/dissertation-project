import logging
import os
import sys

import matplotlib
import numpy as np
import torch
from . import utils
from matplotlib import pyplot as plt
from numba.cuda.testing import numba_cuda_dir
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

matplotlib.use('TkAgg')

class AutoEncoder(nn.Module):
    """
    Auto encoder model for reducing the music samples to a smaller latent space for further clustering and analysis.
    """
    def __init__(self, input_size: int, hidden_layers, dropout_rate: float = 0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size, hidden_layers, dropout_rate)
        self.decoder = Decoder(self.encoder)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_layers, dropout_rate: float = 0.2, activation=nn.LeakyReLU()):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_layers[0])

        self.n_layers = 0
        for i in range(0, len(hidden_layers) - 1):
            setattr(self, f"dense{i}", torch.nn.Linear(self.hidden_layers[i], hidden_layers[i + 1]))
            self.n_layers += 1

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for i in range(0, self.n_layers - 1):
            dense = getattr(self, f"dense{i}")
            x = self.activation(dense(x))
            x = self.dropout(x)

        output_layer = getattr(self, f"dense{self.n_layers - 1}")
        return output_layer(x)


class Decoder(nn.Module):
    def __init__(self, encoder, activation=nn.LeakyReLU()):
        super().__init__()

        self.encoder = encoder
        self.hidden_layers = self.encoder.hidden_layers[::-1]

        for i in range(0, self.encoder.n_layers):
            setattr(self, f"dense{i}", torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))

        self.output_layer = torch.nn.Linear(self.hidden_layers[-1], self.encoder.input_size)
        self.activation = activation
        self.dropout = nn.Dropout(self.encoder.dropout_rate)

    def forward(self, x):
        for i in range(0, self.encoder.n_layers):
            dense = getattr(self, f"dense{i}")
            x = dense(x)
            x = self.activation(x)
            x = self.dropout(x)

        return self.output_layer(x)


class DeepEmbeddedClustering:
    def __init__(self, dataset_loader: utils.Loader, logger: logging.Logger, hidden_layers, dropout_rate: float = 0.2, alpha: float = 1.0, beta: float = 1.0, n_clusters: int = 10):
        self.dataset_loader = dataset_loader
        self.figures_path = self.dataset_loader.get_figures_path()

        self.dataloader = self.dataset_loader.get_dataloader(split_type="train")
        self.input_size = self.dataset_loader.get_input_size()

        self.autoencoder = AutoEncoder(self.input_size, hidden_layers, dropout_rate)

        self.alpha = alpha
        self.beta = beta

        self.loss_vals = []
        self.logger = logger
        self.n_clusters = n_clusters
        self.lr = 0.01
        self.optimiser = torch.optim.SGD(self.autoencoder.parameters(), lr=self.lr, momentum=0.6)
        self.loss_ = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.1)

    def _find_initial_centroids(self, n_clusters: int):
        latent_space = []

        for x_data, y_labels in self.dataloader:
            x_data = x_data.to(self.autoencoder.device)
            embeddings, _ = self.autoencoder(x_data)
            latent_space.extend(embeddings.detach().numpy())

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(latent_space)

        # find the centroids as a tensor flow obj
        centroids = torch.tensor(kmeans.cluster_centers_).to(self.autoencoder.device)
        return centroids

    def train(self, n_epochs: int = 500):
        self.autoencoder.eval()
        best_loss = np.inf

        self.loss_vals = []

        # find some initial centroids to work from
        centroids = self._find_initial_centroids(n_clusters=self.n_clusters)

        tqdm_loop = tqdm(range(n_epochs), desc="Training", unit="epoch")
        for epoch in tqdm_loop:
            losses = []
            current_latent_space = []
            for x_data, y_labels in self.dataloader:
                self.optimiser.zero_grad()

                x_data = x_data.to(self.autoencoder.device)

                embeddings, reconstructed = self.autoencoder(x_data)

                # making sure they are the same data type
                embeddings = torch.tensor(embeddings.detach().numpy(), dtype=torch.float32)
                centroids = torch.tensor(centroids.detach().numpy(), dtype=torch.float32)

                soft_assign = self._soft_assignment(embeddings, centroids)
                targets = self._target_distribution(soft_assign)

                # working out reconstruction loss and kl divergence loss
                kl_loss = self._kl_divergence(targets, soft_assign)
                reconstruction_loss = self.loss_(reconstructed, x_data)

                loss = (kl_loss * self.alpha + reconstruction_loss * self.beta)
                losses.append(loss.item())

                # back-propagation
                loss.backward()

                # this is used to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)

                # update weights
                self.optimiser.step()

                # update new latent space
                current_latent_space.extend(embeddings.detach().numpy())

            mean_loss = np.mean(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss

            tqdm_loop.set_description(f"Training - Loss={mean_loss}")
            self.loss_vals.append(mean_loss)

            # updating centroids
            centroids = self._update_centroids(embeddings=current_latent_space, n_clusters=self.n_clusters)

            self.scheduler.step()

        self.logger.info(f"Training complete! Best Loss={best_loss}")

        return self

    def fit(self):
        self.logger.info("Fitting model")
        data, labels = self.dataset_loader.get_data_split(split_type="test")
        data_tens = torch.tensor(data).to(self.autoencoder.device)

        embeddings = self.autoencoder(data_tens)[0]
        embeddings = embeddings.detach().cpu()

        kmeans = KMeans(n_clusters=self.n_clusters)
        output_labels = kmeans.fit_predict(embeddings)

        output_labels = np.array(output_labels)
        embeddings = np.array(embeddings)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for i in range(self.n_clusters):
            cluster_data = embeddings[output_labels == i]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i}")

        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', label="Centroids", s=200)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        path = os.path.join(self.figures_path, f"dec_clusterings.pdf")
        plt.savefig(path)
        plt.close()

        nmi = normalized_mutual_info_score(labels_true=labels, labels_pred=kmeans.labels_)
        self.logger.info(f"NMI Score: {nmi}")
        # silhouette_avg = silhouette_score(embedded.detach().cpu().numpy(), kmeans.labels_)

        self.logger.info(f"Saved DEC Clustering plot to '{path}'")

    def save(self):
        path = os.path.join(self.dataset_loader.get_directory(), "dec_model")
        torch.save(self.autoencoder, path)

        self.logger.info(f"Saved DEC Model to '{path}'")

        return self

    def plot_loss(self):
        path = os.path.join(self.figures_path, "dec_loss.pdf")
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(1, len(self.loss_vals)+1)], self.loss_vals)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("DEC Loss")
        plt.legend()
        plt.savefig(path)

        self.logger.info(f"Saved Loss plot to '{path}'")

        return self

    def _update_centroids(self, embeddings, n_clusters: int):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)

        # find the centroids as a tensor flow obj
        centroids = torch.tensor(kmeans.cluster_centers_).to(self.autoencoder.device)
        return centroids

    def _kl_divergence(self, targets, assignments):
        """
        Works out the KL Divergence loss between the target labels and their assignments

        :param targets: target values
        :param assignments: assignments
        :return: kl divergence
        """

        return torch.sum(targets * torch.log(targets / (assignments + sys.float_info.epsilon)))

    def _target_distribution(self, assignments):
        """
        This sharpens the predicted probabilities to focuses more confident assignments

        :param assignments: soft assignments
        :return: target distribution
        """

        targets = assignments.pow(2) / (assignments.sum(dim=0, keepdim=True) + 1e-6)
        targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-6)

        return targets

    def _soft_assignment(self, embedded, centroids):
        """
        Use Student's t-Distribution to measure the similarity between the embedded
        points and the centroids.

        :param embedded: latent_space
        :param centroids: cluster centroids
        :return: soft assignments
        """

        q = 1/(1+torch.cdist(embedded, centroids)).pow(2)
        q = q / q.sum(dim=1, keepdim=True)
        return q