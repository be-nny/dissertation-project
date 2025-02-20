import numpy as np
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture

LATENT_DIMS = 2
SEED = 42

class GMMLearner:
    def __init__(self, loader, n_clusters):
        self.n_clusters = n_clusters
        self.loader = loader

        self.umap_model = umap.UMAP(n_components=LATENT_DIMS, n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=SEED)
        self.gaussian_model = GaussianMixture(n_components=self.n_clusters, random_state=SEED, covariance_type='full')

        self.y_true = None
        self.y_pred = None
        self.latent_data = None

    def create_latent(self):
        """
        Creates a latent representation by transforming the data using UMAP, and then applying a Gaussian Mixture Model
        to the data to cluster the latent points.
        """

        tmp = []
        self.y_true = []
        for x, y in self.loader:
            x = x.numpy()
            tmp.extend(x)
            self.y_true.extend(y)

        self.latent_data = self.umap_model.fit_transform(tmp).astype(np.float64)
        self.y_pred = self.gaussian_model.fit_predict(self.latent_data)

        # delete this data to free up memory
        del tmp

        return self

    def fit_new(self, new_data):
        """
        Fit some new data

        :param new_data: 2-D array of flattened data points (n,m) where n is the number of new points, and m is the flattened data.
        :return: latent space, predicted labels
        """

        latent = self.umap_model.fit_transform(new_data).astype(np.float64)
        pred = self.gaussian_model.predict(latent)

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