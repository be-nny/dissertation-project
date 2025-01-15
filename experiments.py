import argparse
import json
import os.path
import numpy as np
import torch
import umap.umap_ as umap
import matplotlib
import config
import logger

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader
from experimental_models import pca, stacked_autoencoder, conv_autoencoder
from model import utils

matplotlib.use('TkAgg')

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - EXPERIMENTS', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-e", "--experiment", type=int, help="Experiment to run")

def show_info(logger, config):
    datasets = os.listdir(config.OUTPUT_PATH)

    for uuid in datasets:
        if uuid[0] != "." and uuid != "experiments":
            path = os.path.join(config.OUTPUT_PATH, uuid)
            with open(os.path.join(path, "receipt.json"), "r") as f:
                data = json.load(f)

                signal_processor = data['preprocessor_info']['signal_processor']
                seg_dur = data['preprocessor_info']['segment_duration']
                total_samples = data['preprocessor_info']['total_samples']
                created_time = data['start_time']

            out_str = f"{uuid} - {signal_processor:<15} SAMPLE SIZE: {total_samples:<5} SEGMENT DURATION:{seg_dur:<5} CREATED:{created_time:<10}"

            logger.info(out_str)

def cluster_info(latent_space, y_true, logger, n_clusters: int = 10):
    # run kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    logger.info(f"KMEANS - NMI: {nmi}")

    # run dbscan
    db_scan = DBSCAN(min_samples=2)
    y_pred = db_scan.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    logger.info(f"DBSCAN - NMI: {nmi}")

    # gaussian model
    gm = GaussianMixture(n_components=10)
    y_pred = gm.fit_predict(latent_space)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    logger.info(f"Gaussian Mixture - NMI: {nmi}")

    # silhouette score
    s = silhouette_score(latent_space, y_true)
    logger.info(f"Silhouette Score: {s}")

def visualise_3D(latent_space, y_true, path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        latent_space[:, 0], latent_space[:, 1], latent_space[:, 2],
        c=y_true, cmap='viridis', alpha=0.7, s=10
    )

    plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    ax.set_title("3D Visualisation of Latent Space")
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_zlabel("Axis 3")
    plt.savefig(path)
    plt.show()

def experiment_5(config, logger, args, experiment_path):
    # load the data
    batch_size = 128
    epochs = 1000
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)

    conv_ae = conv_autoencoder.ConvAutoencoder(
        loader=loader.load(split_type="train", normalise=True),
        logger=logger,
        uuid=args.uuid,
        figures_path=experiment_path,
        epochs=epochs,
        layer_sizes=[loader.input_shape[0], 32, 64, 128]
    )

    conv_ae.train_autoencoder()

def experiment_4(config, logger, args, experiment_path):
    # load the data
    batch_size = 512
    umap_components = 64
    latent_dims = 4

    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)
    train_data = []
    train_labels = []
    for x, y in loader.load(split_type="train", normalise=True):
        for val in x:
            train_data.append(val.detach().cpu().numpy())
        train_labels.extend(y)
    train_data = np.array(train_data)
    train_data = train_data.reshape(train_data.shape[0], -1)
    train_labels = np.array(train_labels)

    # run UMAP on input signals
    mapper = umap.UMAP(min_dist=0.0, spread=3, n_components=umap_components, n_neighbors=15, local_connectivity=3)
    train_umap_space = mapper.fit_transform(train_data)

    # create torch dataset loader
    data_tensor = torch.tensor(np.array(train_umap_space), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(train_labels), dtype=torch.int64)
    tensor_dataset = TensorDataset(data_tensor, labels_tensor)
    train_data_loader = DataLoader(tensor_dataset, batch_size=batch_size)

    # parsing this into an autoencoder
    hidden_layers = [umap_components, 512, 256, 16, latent_dims]
    sae = stacked_autoencoder.SAE(
        hidden_layers=hidden_layers,
        logger=logger,
        loader=train_data_loader,
        uuid=args.uuid,
        epochs=1000,
        dropout_rate=0.2,
        figures_path=experiment_path,
    )
    sae.train_autoencoder()

    # visuals the embeddings
    # load the test data
    test_data = []
    test_labels = []
    for x, y in loader.load(split_type="test", normalise=True):
        for val in x:
            test_data.append(val.detach().cpu().numpy())
        test_labels.extend(y)
    test_data = np.array(test_data)
    test_data = test_data.reshape(test_data.shape[0], -1)
    test_labels = np.array(test_labels)

    # use the mapper to map it using UMAP
    test_umap_space = mapper.fit_transform(test_data)

    # create the data loaders
    data_tensor = torch.tensor(np.array(test_umap_space), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.int64)
    tensor_dataset = TensorDataset(data_tensor, labels_tensor)
    test_data_loader = DataLoader(tensor_dataset, batch_size=batch_size)

    # fit this new data
    latent_space = []
    train_labels = []
    sae.eval()
    for x, y in test_data_loader:
        x = x.to(sae.device)
        l, _ = sae(x)
        latent_space.extend(l.detach().cpu().numpy())
        train_labels.extend(y)

    latent_space = np.array(latent_space)

    # cluster
    cluster_info(y_true=train_labels, logger=logger, n_clusters=10, latent_space=latent_space)

    # plot results
    path = os.path.join(experiment_path, f"{args.uuid}_umap_sae_visualisation.pdf")
    visualise_3D(latent_space, train_labels, path)

def experiment_3(config, logger, args, experiment_path):
    # load the data
    batch_size = 512
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)
    data = []
    y_true = []
    for x, y in loader.load(split_type="all", normalise=True):
        for val in x:
            data.append(val.detach().cpu().numpy())
        y_true.extend(y)
    data = np.array(data)
    reshaped_data = data.reshape(data.shape[0], -1)
    y_true = np.array(y_true)

    # run UMAP on input signals
    latent_space = umap.UMAP(min_dist=0.0, spread=3, n_components=3, n_neighbors=15, local_connectivity=3).fit_transform(reshaped_data)

    # cluster scores
    cluster_info(y_true=y_true, logger=logger, n_clusters=10, latent_space=latent_space)

    # plot results
    path = os.path.join(experiment_path, f"{args.uuid}_umap_visualisation.pdf")
    visualise_3D(latent_space, y_true, path)

def experiment_2(config, logger, args, experiment_path):
    batch_size = 512
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)

    # creating pca model
    pca_components = 50
    latent_dims = 5
    pca_model = pca.PCAModel(logger=logger, output_path=experiment_path,
                             n_components=pca_components, uuid=args.uuid)
    data = []
    labels = []

    for x, y in loader.load(split_type="train", normalise=True):
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
    sae = stacked_autoencoder.SAE(
        hidden_layers=hidden_layers,
        logger=logger,
        loader=reduced_loader,
        epochs=1000,
        dropout_rate=0.1,
        uuid=args.uuid,
        figures_path=experiment_path,
    )
    sae.train_autoencoder()

    # visuals the embeddings
    # load the test data
    test_data = []
    test_labels = []
    for x, y in loader.load(split_type="test", normalise=True):
        for val in x:
            test_data.append(val.detach().cpu().numpy())
        test_labels.extend(y)
    test_data = np.array(test_data)
    test_data = test_data.reshape(test_data.shape[0], -1)
    test_labels = np.array(test_labels)

    # use the mapper to map it using UMAP
    test_umap_space = pca_model.fit_transform(test_data)

    # create the data loaders
    data_tensor = torch.tensor(np.array(test_umap_space), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.int64)
    tensor_dataset = TensorDataset(data_tensor, labels_tensor)
    test_data_loader = DataLoader(tensor_dataset, batch_size=batch_size)

    # fit this new data
    latent_space = []
    train_labels = []
    sae.eval()
    for x, y in test_data_loader:
        x = x.to(sae.device)
        l, _ = sae(x)
        latent_space.extend(l.detach().cpu().numpy())
        train_labels.extend(y)

    latent_space = np.array(latent_space)

    # cluster
    cluster_info(y_true=train_labels, logger=logger, n_clusters=10, latent_space=latent_space)

    # plot results
    path = os.path.join(experiment_path, f"{args.uuid}_pca_sae_visualisation.pdf")
    visualise_3D(latent_space, train_labels, path)

def experiment_1(config, logger, args, experiment_path):
    batch_size = 512
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=batch_size)

    # creating pca model
    pca_components = 3
    pca_model = pca.PCAModel(logger=logger, output_path=experiment_path, n_components=pca_components, uuid=args.uuid)
    data = []
    y_true = []

    for x, y in loader.load(split_type="all", normalise=False):
        x = x.numpy()
        flattened = [i.flatten() for i in x]
        data.extend(flattened)
        y_true.extend(y)

    latent_space = pca_model.fit_transform(data)
    pca_model.plot_eigenvalues()

    # cluster
    cluster_info(y_true=y_true, logger=logger, n_clusters=10, latent_space=latent_space)

    # plot results
    path = os.path.join(experiment_path, f"{args.uuid}_pca_visualisation.pdf")
    visualise_3D(latent_space, y_true, path)

if __name__ == "__main__":
    args = parser.parse_args()

    config = config.Config(path=args.config)
    logger = logger.get_logger()

    #
    experiments_dir = os.path.join(config.OUTPUT_PATH, "experiments")
    if not os.path.exists(experiments_dir):
        os.mkdir(experiments_dir)

    if args.info:
        show_info(logger, config)

    if args.uuid:
        if args.experiment == 1:
            exp_1_path = os.path.join(experiments_dir, "experiment_1")
            if not os.path.exists(exp_1_path):
                os.mkdir(exp_1_path)

            logger.info("Running Experiment 1")
            experiment_1(config, logger, args, exp_1_path)
        if args.experiment == 2:
            exp_2_path = os.path.join(experiments_dir, "experiment_2")
            if not os.path.exists(exp_2_path):
                os.mkdir(exp_2_path)

            logger.info("Running Experiment 2")
            experiment_2(config, logger, args, exp_2_path)
        if args.experiment == 3:
            exp_3_path = os.path.join(experiments_dir, "experiment_3")
            if not os.path.exists(exp_3_path):
                os.mkdir(exp_3_path)

            logger.info("Running Experiment 3")
            experiment_3(config, logger, args, exp_3_path)
        if args.experiment == 4:
            exp_4_path = os.path.join(experiments_dir, "experiment_4")
            if not os.path.exists(exp_4_path):
                os.mkdir(exp_4_path)

            logger.info("Running Experiment 4")
            experiment_4(config, logger, args, exp_4_path)

        if args.experiment == 5:
            exp_5_path = os.path.join(experiments_dir, "experiment_5")
            if not os.path.exists(exp_5_path):
                os.mkdir(exp_5_path)

            logger.info("Running Experiment 5")
            experiment_5(config, logger, args, exp_5_path)
