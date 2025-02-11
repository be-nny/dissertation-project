import numpy as np
import umap.umap_ as umap
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from tqdm import tqdm

LATENT_DIMS = 2
SEED = 42

class MetricLearner:
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

        data = []
        self.y_true = []
        for x, y in self.loader:
            x = x.numpy()
            data.extend(x)
            self.y_true.extend(y)

        self.latent_data = self.umap_model.fit_transform(data).astype(np.float64)
        self.y_pred = self.gaussian_model.fit_predict(self.latent_data)

        # delete this data to free up memory
        del data

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

class GenreClassifier(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list, output_dims: int):
        dims = [input_dims, *hidden_dims, output_dims]

        self.network = []
        for i in range(0, len(dims)-1):
            self.network.extend([nn.Linear(dims[i], dims[i+1]), nn.Sigmoid()])
        self.network.append(nn.LogSoftmax(dim=1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Feed forward through the network

        :param x: input tensor
        :return: the output of the network
        """

        for f in self.network:
            x = f(x)
        return x

def train_genre_classifier(model: GenreClassifier, n_epochs: int, loader, lr: float = 0.1):
    """
    Trains a multi-classifier to classify genres.

    :param model: genre classifier
    :param n_epochs: number of epochs
    :param loader: dataset loader
    """

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    tqdm_loop = tqdm(range(n_epochs), desc="Training Genre Classifier", unit="epoch")
    for _ in tqdm_loop:
        epoch_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            x = x.to(model.device)

            y_hat_log = model(x)
            loss = criterion(y_hat_log, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        tqdm_loop.set_description(f"Training Genre Classifier - avg_loss:{avg_loss}")



