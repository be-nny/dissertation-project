import logging
import os
import uuid

import matplotlib
import networkx as nx
import numpy as np
import umap.umap_ as umap

from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA

matplotlib.use('TkAgg')

class PCAModel(IncrementalPCA):
    def __init__(self, logger: logging.Logger, data: list, genre_labels: list, figures_path: str, batch_size=8, n_components=None):
        if n_components is not None:
            super().__init__(batch_size=batch_size, n_components=n_components)
        else:
            super().__init__(batch_size=batch_size)

        self.logger = logger
        self.figures_path = figures_path

        self.data, self.genre_labels = data, genre_labels
        self.embeddings = None

    def create(self):
        """
        Fits the data using PCA via `partial_fit` to batch load the data. This is saved
        in the 'figures/' directory in the UUID path provided.

        :return: Current instance
        """

        self.logger.info("Generating PCA Model...")
        self.partial_fit(self.data)
        self.embeddings = self.transform(self.data)

        return self

    def get_embeddings(self):
        """
        :return: The embeddings after PCA has been applied
        """

        self.embeddings = np.clip(self.embeddings, -1e10, 1e10)

        return self.embeddings

    def visualise(self, path=None):
        """
        Plots the eigenvalues on a logarithmic scale agains the number of PCA components.
        """
        if path is None:
            path = os.path.join(self.figures_path, "eigenvalues.pdf")

        plt.figure(figsize=(10, 10))
        plt.plot(range(1, len(self.explained_variance_) + 1), self.explained_variance_, marker='o', linestyle='--')
        plt.yscale('log')
        plt.title('PCA Eigenvalues (Log Scale)')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue (Log Scale)')
        plt.grid(True)
        plt.savefig(path)
        plt.close()
        self.logger.info(f"saved '{path}'")


class UmapModel(umap.UMAP):
    def __init__(self, data: list, genre_labels: list, logger: logging.Logger, figures_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        self.data = data
        self.genre_labels = genre_labels
        self.embeddings = None
        self.logger = logger
        self.figures_path = figures_path

    def create(self):
        """
        Fits the data using UMAP.
        :return: Current object
        """

        self.logger.info("Generating UMAP Model...")
        self.embeddings = self.fit_transform(self.data)

        return self

    def visualise(self, path=None):
        """
        Plots a Graph using UMAPs underlying knn graph to understand the relationship between the
        features in the dataset. This is saved in the 'figures/' directory in the UUID path provided.
        """
        id = str(uuid.uuid4().hex)[:6]

        if path is None:
            path = os.path.join(self.figures_path, f"umap_{id}.pdf")

        G = nx.Graph(self.graph_)
        plt.figure(figsize=(8, 8))
        plt.title(', '.join(str(f"{k}:{v}") for k, v in self.kwargs.items()))

        nx.draw(G, node_size=10, alpha=0.5)
        plt.savefig(path)
        plt.close()
        self.logger.info(f"saved '{path}'")

    def get_embeddings(self):
        return self.embeddings

