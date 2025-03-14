import config
import matplotlib
import numpy as np
import model
import logger

from model import models, utils
from plot_lib import plotter

matplotlib.use('TkAgg')

config = config.Config(path="config.yml")
logger = logger.get_logger()
loader = utils.Loader(out=config.OUTPUT_PATH, uuid="19ee37", logger=logger, batch_size=model.BATCH_SIZE)
loader.load(split_type="all", normalise=True, flatten=True)

convex_cluster_model = models.ConvexCluster(loader=loader)
lambda_values = np.logspace(-2, 1, 30)
u_path = convex_cluster_model.convex_cluster(lambda_vals=lambda_values, k=50)

plotter.plot_conex_clusters(latent_space=convex_cluster_model.latent_space, u_path=u_path, y_true=convex_cluster_model.y_true, loader=loader)