import argparse
import json
import os.path

import numpy as np
import torch
from pyarrow import show_info
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader

import config
from model import deep_clustering, utils, pca, autoencoder
import logger

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - DEEP EMBEDDED CLUSTERING MODEL', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-e", "--experiment", type=int,required=True, help="Experiment to run")

def show_info(logger, config):
    datasets = os.listdir(config.OUTPUT_PATH)

    for uuid in datasets:
        if uuid[0] != ".":
            path = os.path.join(config.OUTPUT_PATH, uuid)
            with open(os.path.join(path, "receipt.json"), "r") as f:
                data = json.load(f)

                signal_processor = data['preprocessor_info']['signal_processor']
                seg_dur = data['preprocessor_info']['segment_duration']
                total_samples = data['preprocessor_info']['total_samples']
                created_time = data['start_time']

            out_str = f"{uuid} - {signal_processor:<15} SAMPLE SIZE: {total_samples:<5} SEGMENT DURATION:{seg_dur:<5} CREATED:{created_time:<10}"

            logger.info(out_str)

def experiment_1(config, logger, args):
    batch_size = 512
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)

    # creating pca model
    pca_components = 50
    latent_dims = 5
    pca_model = pca.PCAModel(logger=logger, uuid_path=os.path.join(config.OUTPUT_PATH, args.uuid),
                             n_components=pca_components)
    data = []
    labels = []

    for x, y in loader.load(split_type="train"):
        x = x.numpy()
        flattened = [i.flatten() for i in x]
        data.extend(flattened)
        labels.extend(y)

    x_reduced = pca_model.fit_transform(data)
    pca_model.plot_eigenvalues()

    # create torch dataset loader
    data_tensor = torch.tensor(np.array(x_reduced), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.int64)
    reduced_dataset = TensorDataset(data_tensor, labels_tensor)
    reduced_loader = DataLoader(reduced_dataset, batch_size=batch_size)

    # parsing this into an autoencoder
    hidden_layers = [pca_components, 2048, 1024, 128, latent_dims]
    ae = autoencoder.AutoEncoder(hidden_layers=hidden_layers, logger=logger, loader=reduced_loader, epochs=10000,
                                 dropout_rate=0.1, figures_path=loader.get_figures_path())
    ae.train()

    model = ae.model()
    latent_space = []
    y_true = []
    model.eval()
    for x, y in reduced_loader:
        x = x.to(ae.device)
        l, _ = model(x)
        latent_space.extend(l.detach().cpu().numpy())
        y_true.extend(y)

    # run kmeans
    kmeans = KMeans(n_clusters=10)
    y_pred = kmeans.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    s = silhouette_score(latent_space, y_true)
    logger.info(f"KMEANS - NMI: {nmi}, Silhouette Score: {s}")

    # run dbscan
    db_scan = DBSCAN(min_samples=2)
    y_pred = db_scan.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    s = silhouette_score(latent_space, y_true)
    logger.info(f"DBSCAN - NMI: {nmi}, Silhouette Score: {s}")

    # gaussian model
    gm = GaussianMixture(n_components=10)
    y_pred = gm.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    s = silhouette_score(latent_space, y_true)
    logger.info(f"Gaussian Mixture - NMI: {nmi}, Silhouette Score: {s}")

def experiment_2(config, logger, args):
    pass

if __name__ == "__main__":
    args = parser.parse_args()

    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)

    if args.uuid:
        if args.experiment == 1:
            logger.info("Running Experiment 1")
            experiment_1(config, logger, args)
        if args.experiment == 2:
            logger.info("Running Experiment 2")
            experiment_2(config, logger, args)

