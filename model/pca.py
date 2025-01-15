import os.path

import matplotlib

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

class PCAModel(PCA):
    def __init__(self, output_path, uuid, logger, n_components=100):
        super().__init__(n_components=n_components)
        self.output_path = output_path
        self.logger = logger
        self.uuid = uuid

    def plot_eigenvalues(self):
        path = os.path.join(self.output_path, f"{self.uuid}_eigenvalues.pdf")

        plt.plot([i for i in range(1, self.n_components + 1)], self.explained_variance_, marker="o", linestyle="-", label="Eigenvalues")
        plt.xlabel("Number of Components")
        plt.ylabel("Eigenvalues (log)")
        plt.yscale("log")

        plt.title("PCA Eigenvalues (log scale)")
        plt.savefig(path)
        self.logger.info(f"Saved plot '{path}'")
        plt.close()
