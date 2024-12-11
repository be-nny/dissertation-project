import logging
import os
import matplotlib
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA
from . import utils

matplotlib.use('Agg')


class PCAModel:
    def __init__(self, out: str, uuid: str, n_components, loader: utils.Loader, logger: logging.Logger):
        self.out = out
        self.uuid = uuid
        self.path = os.path.join(self.out, self.uuid)
        self.loader = loader
        self.logger = logger

        self.ipca = IncrementalPCA(n_components=n_components)
        self.embeddings = []

    def create(self):
        self.logger.info("Creating PCA model...")

        # fitting
        generator = self.loader.get_batch("train", batch_size=25)
        for data, genre in tqdm(generator, desc="Fitting", unit="batch"):
            self.ipca.partial_fit(data)

        # transforming
        generator = self.loader.get_batch("test", batch_size=25)
        self.embeddings.extend(self.ipca.transform(data) for data, _ in tqdm(generator, desc="Transforming (test)", unit="batch"))

        generator = self.loader.get_batch("train", batch_size=25)
        self.embeddings.extend(self.ipca.transform(data) for data, _ in tqdm(generator, desc="Transforming (train)", unit="batch"))

        self.embeddings = np.vstack(self.embeddings)

    def plot(self):
        self.logger.info("Plotting PCA model...")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.embeddings[:, 0], self.embeddings[:, 1], self.embeddings[:, 2], c='blue', edgecolors='k',
                   alpha=0.7)
        ax.set_title("PCA: 3 Principal Components")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()
