import logging
import os
import matplotlib
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from . import utils

matplotlib.use('Agg')


class PCAModel:
    def __init__(self, out: str, uuid: str, n_components, loader: utils.Loader, logger: logging.Logger):
        self.out = out
        self.uuid = uuid
        self.path = os.path.join(self.out, self.uuid)
        self.loader = loader
        self.logger = logger

        self.pca = PCA()
        self.embeddings = []

    def create(self):
        self.logger.info("Creating PCA model...")
        data = self.loader.get_data(split_type='train')
        data.extend(self.loader.get_data(split_type='test'))

        self.pca.fit(data)
