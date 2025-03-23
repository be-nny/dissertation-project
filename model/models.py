import umap.umap_ as umap
import numpy as np
import cvxpy as cp

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
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
        self.centres = []
        self.clustering_path = []
        self.y_pred = None

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

            self.clustering_path.append(cluster_centre.value)

        self.centres = np.array(self.clustering_path[-1])
        self.y_pred = self._create_labels()

        return self.clustering_path

    def _create_labels(self, tol=1e-3):
        n = self.centres.shape[0]
        labels = -np.ones(n, dtype=np.int64)
        current_label = 0
        for i in range(n):
            if labels[i] == -1:
                for j in range(i+1,n):
                    if labels[j] == -1 and np.linalg.norm(self.centres[i] - self.centres[j]) < tol:
                        labels[j] = current_label
                current_label +=1
        return labels