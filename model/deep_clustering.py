import logging
import os
import matplotlib
import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans

from . import utils
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

matplotlib.use('TkAgg')

class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_layers, dropout_rate: float = 0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size, hidden_layers, dropout_rate, activation=nn.ELU())
        self.decoder = Decoder(self.encoder, activation=nn.ELU())
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

class DEC(nn.Module):
    def __init__(self, autoencoder, latent_space_dims: int, n_clusters: int = 10):
        super().__init__()

        self.autoencoder = autoencoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a clustering layer
        # this is the learnable parameter
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_space_dims))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)

    def forward(self, x):
        latent_space, reconstructed = self.autoencoder(x)
        q = self._t_distribution(latent_space, self.clustering_layer)

        return q, latent_space, reconstructed

    def _t_distribution(self, embedded, centroids):
        """
        Use Student's t-Distribution to create a vector of probabilities of this point being assigned to a particular
        cluster. This vector is then normalised.

        :param embedded: latent_space
        :param centroids: cluster centroids
        :return: soft assignments (num_samples, num_clusters)
        """

        soft_assign = 1/(1+torch.cdist(embedded, centroids).pow(2))
        soft_assign = soft_assign / soft_assign.sum(dim=1, keepdim=True)

        return soft_assign

class ClusteringModel:
    def __init__(self, dataset_loader: utils.Loader, logger: logging.Logger, hidden_layers, dropout_rate: float = 0.2, n_clusters: int = 10, pre_train_epochs: int = 500):
        self.dataset_loader = dataset_loader
        self.figures_path = self.dataset_loader.get_figures_path()
        self.dataloader = self.dataset_loader.get_dataloader(split_type="train", batch_size=128)
        self.input_size = self.dataset_loader.get_input_size()
        self.logger = logger

        self.n_clusters = n_clusters

        # autoencoder
        self.autoencoder = AutoEncoder(self.input_size, hidden_layers, dropout_rate)
        self.ae_lr = 0.001
        self.ae_optimiser = torch.optim.AdamW(self.autoencoder.parameters(), lr=self.ae_lr)
        self.ae_scheduler = lr_scheduler.StepLR(self.ae_optimiser, step_size=100, gamma=0.1)

        # pretrain the autoencoder
        self._pre_train(n_epochs=pre_train_epochs)

        # dec
        self.dec = DEC(self.autoencoder, hidden_layers[-1], n_clusters)
        self.dec_lr = 0.0001
        self.dec_optimiser = torch.optim.Adam(self.dec.parameters(), lr=self.dec_lr)
        self.dec_scheduler = lr_scheduler.StepLR(self.dec_optimiser, step_size=100, gamma=0.1)

    def _pre_train(self, n_epochs: int = 500):
        self.autoencoder.train()

        recon_loss_fn = nn.MSELoss()
        total_losses = []
        best_loss = np.inf

        tqdm_loop = tqdm(range(n_epochs), desc="Pre-training Autoencoder", unit="epoch")
        for _ in tqdm_loop:
            losses = []
            for x_data, y_labels in self.dataloader:
                self.ae_optimiser.zero_grad()
                x_data = x_data.to(self.autoencoder.device)
                embeddings, reconstructed = self.autoencoder(x_data)

                reconstruction_loss = recon_loss_fn(reconstructed, x_data)
                losses.append(reconstruction_loss.item())
                reconstruction_loss.backward()

                # prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)

                self.ae_optimiser.step()

            mean_loss = np.mean(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss

            tqdm_loop.set_description(f"Pre-training Autoencoder - Loss={mean_loss}")
            total_losses.append(mean_loss)

            self.ae_scheduler.step()

        self.logger.info(f"Pre-training Autoencoder complete! Best Loss={best_loss}")

        # plot pre-training loss
        self._plot_figure(title="Pre-training Loss", path=os.path.join(self.figures_path, "pre-training_loss.pdf"), epochs=[i for i in range(1, len(total_losses) + 1)], loss=total_losses)
        return self

    def train(self, clustering_epochs: int = 500, update_freq: int = 5):
        self.dec.train()

        # set the loss functions to use
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        recon_loss_fn = nn.MSELoss()

        total_losses = []
        total_nmi = []
        total_ari = []

        # initialise the centroids
        kmeans = KMeans(n_clusters=self.n_clusters)
        total_data = []
        total_labels = []
        for x, y in self.dataloader:
            total_data.extend(x)
            total_labels.extend(y)
        total_data = torch.tensor(np.array(total_data), dtype=torch.float32).to(self.autoencoder.device)
        latent_space, _ = self.autoencoder(total_data)
        y_pred = kmeans.fit_predict(latent_space.detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        self.dec.clustering_layer.data = torch.tensor(cluster_centers, dtype=torch.float32).to(self.dec.device)
        y_prev = y_pred

        best_loss = np.inf
        tqdm_loop = tqdm(range(clustering_epochs), desc="Training", unit="epoch")
        for epoch in tqdm_loop:
            losses = []

            # updating cluster centers
            if epoch % update_freq == 0:
                q, _, _ = self.dec(total_data)
                p = self._target_distribution(q)

                # pick the largest probability from the t distribution, this will be the predicted label
                # delta_label is the fraction of labels that have switched assignments - lower means more convergence
                y_pred = q.detach().cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_prev).astype(np.float32) / y_prev.shape[0]

                nmi = sklearn.metrics.normalized_mutual_info_score(total_labels, y_pred)
                ari = sklearn.metrics.adjusted_rand_score(total_labels, y_pred)
                y_prev = y_pred

                total_nmi.append(nmi)
                total_ari.append(ari)

            st_ptn = 0
            end_ptn = self.dataloader.batch_size
            for x_data, y_true in self.dataloader:
                self.dec_optimiser.zero_grad()
                x_data = x_data.to(self.dec.device)
                q, latent_space, reconstructed = self.dec(x_data)

                # work out loss
                kl_loss = kl_loss_fn(torch.log(q + 1e-10), torch.tensor(p.detach().cpu().numpy()[st_ptn:end_ptn]).to(self.dec.device))
                reconstruction_loss = recon_loss_fn(reconstructed, x_data)
                loss = kl_loss * 3 + reconstruction_loss

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
        self._plot_figure(title="Normalised Mutual Information Scores", path=os.path.join(self.figures_path, "nmi_scores.pdf"), epochs=[i for i in range(1, len(total_nmi) + 1, update_freq)], nmi=total_nmi)
        self._plot_figure(title="Adjusted Rand Index Scores", path=os.path.join(self.figures_path, "ari_scores.pdf"), epochs=[i for i in range(1, len(total_ari) + 1, update_freq)], nmi=total_ari)

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
