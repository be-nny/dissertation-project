import umap.umap_ as umap
import numpy as np
import cvxpy as cp
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from torch import nn
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from model import utils

LATENT_DIMS = 2
SEED = 42

def _get_cluster_type(cluster_type: str, n_clusters: int):
    if cluster_type == 'kmeans':
        return KMeans(random_state=SEED, n_clusters=n_clusters)
    elif cluster_type == 'gmm':
        return GaussianMixture(n_components=n_clusters, random_state=SEED, covariance_type='full')

def get_dim_model(model_type):
    if model_type.lower() == "pca":
        return PCA(n_components=LATENT_DIMS, random_state=SEED)
    elif model_type == "umap":
        return umap.UMAP(n_components=LATENT_DIMS, n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=SEED)
    elif model_type == "tsne":
        return TSNE(n_components=LATENT_DIMS, random_state=SEED)
    else:
        raise TypeError("Model type must be 'pca' or 'umap' or 'tsne'")

class MetricLeaner:
    def __init__(self, loader: utils.Loader, n_clusters: int, cluster_type: str):
        self.loader = loader
        self.n_clusters = n_clusters

        self.cluster_model = _get_cluster_type(cluster_type, n_clusters)
        self.dim_reducer = get_dim_model("umap")

        self.y_true = None
        self.y_pred = None
        self.latent_data = None

        self.data_points = None

        self._create_latent()

    def _create_latent(self):
        """
        Creates a latent representation by transforming the data using UMAP, and then applying a Gaussian Mixture Model
        to the data to cluster the latent points.
        """

        tmp, self.y_true = self.loader.all()
        self.latent_data = self.dim_reducer.fit_transform(tmp).astype(np.float64)
        self.y_pred = self.cluster_model.fit_predict(self.latent_data)

        # delete this data to free up memory
        del tmp

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


class ConvexCluster:
    def __init__(self, loader: utils.Loader):
        self.loader = loader
        self.dim_reducer = get_dim_model("umap")

        self._create_latent()

    def _create_latent(self):
        tmp, self.y_true = self.loader.all()
        self.latent_space = self.dim_reducer.fit_transform(tmp).astype(np.float64)
        del tmp

    def convex_cluster(self, lambda_vals, k):
        """
        Minimises a penalising loss function over a range of lambda values to show the evolution of the cluster centres.

        :param lambda_vals: lambda vals
        :param k: k nearest neighbours
        :return:
        """
        clustering_path = []
        n, m = self.latent_space.shape
        dists = euclidean_distances(self.latent_space)
        np.fill_diagonal(dists, np.inf)

        # initialise the weights by finding the k nearest neighbours to each point
        weights = np.zeros((n, n))
        for i in range(n):
            nearest = np.argsort(dists[i])[:k]
            weights[i, nearest] = np.exp(-dists[i, nearest] ** 2)

        tqdm_loop = tqdm(lambda_vals, desc="Clustering...")
        for lambda_val in tqdm_loop:
            cluster_centre = cp.Variable((n, m))

            # the loss function
            loss_func = 0.5 * cp.sum_squares(self.latent_space - cluster_centre)

            # the penalisation function
            # these calculations have been vectorised for performance opt.
            diff = cp.reshape(cluster_centre, (n, 1, m), order="F") - cp.reshape(cluster_centre, (1, n, m), order="F")
            diff_flat = cp.reshape(diff, (n * n, m), order="F")
            norms_flat = cp.norm(diff_flat, 2, axis=1)
            norms = cp.reshape(norms_flat, (n, n), order="F")
            penalty_func = lambda_val * 0.5 * cp.sum(cp.multiply(weights, norms))

            # minimise this loss function
            minimise_pen_loss_func = cp.Problem(cp.Minimize(loss_func + penalty_func))
            minimise_pen_loss_func.solve()
            clustering_path.append(cluster_centre.value)

        return clustering_path


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

