import umap.umap_ as umap
import numpy as np
import torch
from sklearn.cluster import KMeans

from torch import nn
from sklearn.mixture import GaussianMixture
from model import utils

LATENT_DIMS = 2
SEED = 42

def _get_cluster_type(cluster_type: str, n_clusters: int):
    if cluster_type == 'kmeans':
        return KMeans(random_state=SEED, n_clusters=n_clusters)
    elif cluster_type == 'gmm':
        return GaussianMixture(n_components=n_clusters, random_state=SEED, covariance_type='full')

class MetricLeaner:
    def __init__(self, loader: utils.Loader, n_clusters: int, cluster_type: str):
        self.loader = loader
        self.n_clusters = n_clusters

        self.cluster_model = _get_cluster_type(cluster_type, n_clusters)
        self.dim_reducer = umap.UMAP(n_components=LATENT_DIMS, n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=SEED)

        self.y_true = None
        self.y_pred = None
        self.latent_data = None

        self.data_points = None

    def create_latent(self):
        """
        Creates a latent representation by transforming the data using UMAP, and then applying a Gaussian Mixture Model
        to the data to cluster the latent points.
        """

        tmp, self.y_true = self.loader.all()
        self.latent_data = self.dim_reducer.fit_transform(tmp).astype(np.float64)
        self.y_pred = self.cluster_model.fit_predict(self.latent_data)

        # delete this data to free up memory
        del tmp

        return self

    def fit_new(self, new_data):
        """
        Fit some new data

        :param new_data: 2-D array of flattened data points (n,m) where n is the number of new points, and m is the flattened data.
        :return: latent space, predicted labels
        """

        latent = self.dim_reducer.fit_transform(new_data).astype(np.float64)
        pred = self.cluster_model.predict(latent)

        return latent, pred

    def get_latent(self):
        """
        :return: the latent representation of the data
        """
        return self.latent_data

    def get_y_pred(self):
        """
        :return: the predicted labels
        """
        return self.y_pred

    def get_y_true(self):
        """
        :return: the true labels
        """
        return self.y_true

def target_distribution(assignments):
    """
    This sharpens the predicted probabilities to focuses more confident assignments
    :param assignments: soft assignments
    :return: target distribution (num_samples, num_clusters)
    """

    targets = assignments.pow(2) / (assignments.sum(dim=0, keepdim=True) + 1e-6)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return targets

class DEC(nn.Module):
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

class Conv1DAutoencoder(nn.Module):
    def __init__(self, n_layers: list, input_shape, latent_dim=10):
        super().__init__()
        self.n_layers = n_layers
        K_S = 3
        S = 2
        P = 1

        # encoder
        encoder_layers = []
        self.current_length = input_shape[1]
        for i in range(0, len(n_layers) - 1):
            # work out the size after conv and max pool is applied
            self.current_length = (self.current_length + 2 * P - K_S) // S + 1
            self.current_length = (self.current_length + 2 * P - K_S) // S + 1

            encoder_layers.append(nn.Conv1d(n_layers[i], n_layers[i + 1], kernel_size=K_S, stride=S, padding=P))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.MaxPool1d(kernel_size=K_S, stride=S, padding=P))

        # adding FCN
        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(self.current_length*n_layers[-1], latent_dim))
        encoder_layers.append(nn.LeakyReLU())

        # decoder
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
        latent = self.encoder(x)

        linear = self.linear_decoder(latent)
        linear_reshaped = linear.view(linear.size(0), self.n_layers[-1], self.current_length)

        decoded = self.decoder(linear_reshaped)

        return latent, decoded

