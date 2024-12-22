import os.path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

import config
from mat_logger import mat_logger
from mat_models import deep_clustering, utils

config = config.Config(path="config.yml")
logger = mat_logger.get_logger()

# loading all data (combining test and train splits together)
dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="d1cf0c", logger=logger)

# deep embedded clustering model
dec_model = deep_clustering.DeepEmbeddedClustering(dataset_loader, logger=logger, hidden_layers=[2048, 1024, 256, 64, 10, 2], alpha=1.4, beta=1)
dec_model.train(n_epochs=100)
dec_model.plot_loss()

# fitting model
dec_model.fit()
dec_model.save()

