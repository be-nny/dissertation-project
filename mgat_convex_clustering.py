import argparse
import os.path

import config
import matplotlib
import numpy as np
import model
import logger

from model import models, utils
from plot_lib import plotter

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(prog='Music Genre Analysis Tool - CONVEX CLUSTER', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use, or a list of comma seperated uuid's")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    root = f"{config.OUTPUT_PATH}/_convex_clustering/"
    if not os.path.exists(root):
        os.mkdir(root)

    logger = logger.get_logger()
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
    loader.load(split_type="all", normalise=True, flatten=True)

    convex_cluster_model = models.ConvexCluster(loader=loader)
    lambda_values = np.logspace(-2, 1, 30)
    u_path = convex_cluster_model.convex_cluster(lambda_vals=lambda_values, k=50)

    path = os.path.join(root, f"convex_clustering_{args.uuid}.pdf")
    plotter.plot_conex_clusters(latent_space=convex_cluster_model.latent_space, u_path=u_path, y_true=convex_cluster_model.y_true, loader=loader, path=path)
    logger.info(f"Saved plot '{path}'")
