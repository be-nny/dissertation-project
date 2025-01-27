import os
import argparse
import os.path
import numpy as np
import torch
import umap.umap_ as umap
import matplotlib

from kneed import KneeLocator
from matplotlib.colors import ListedColormap
from torch.utils.data import TensorDataset, DataLoader

import config
import logger

from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from model import utils
from experimental_models import stacked_autoencoder

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - EXPERIMENTS', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-n", "--nmi", type=str, help="Plots NMI score against the number of UMAP or PCA components. Set flag to 'pca' or 'umap'")
parser.add_argument("-e", "--eigen", type=int, help="Plots the eigenvalues obtained after performing PCA. Takes value for max n_components")
parser.add_argument("-b", "--boundaries", help="Plots 2D Kmeans Boundaries of UMAP or PCA. '-nc' must be set for the number of clusters. '-g' must also be set.")
parser.add_argument("-t", "--inertia", help="Plots number of clusters against kmeans inertia score for UMAP or PCA. '-g' must also be set. '-nc' must be set for the max. number of clusters")
parser.add_argument("-v2", "--visualise2d", help="Plots 2D Latent Space of UMAP or PCA. '-g' must also be set.")
parser.add_argument("-v3", "--visualise3d", help="Plots 3D Latent Space of UMAP or PCA. '-g' must also be set.")
parser.add_argument("-sae", "--stackedae", action="store_true", help="Plots 2D Latent Space using a stacked auto encoder. '-g' must also be set.")
parser.add_argument("-nc", "--n_clusters", type=int, help="number of clusters")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to all, all genres are used")

BATCH_SIZE = 512

def show_info(logger, config):
    datasets = os.listdir(config.OUTPUT_PATH)

    for uuid in datasets:
        if uuid[0] != "." and uuid != "experiments":
            path = os.path.join(config.OUTPUT_PATH, uuid)
            receipt_file = os.path.join(path, "receipt.json")
            with utils.ReceiptReader(filename=receipt_file) as receipt_reader:
                out_str = f"{uuid} - {receipt_reader.signal_processor:<15} SAMPLE SIZE: {receipt_reader.total_samples:<5} SEGMENT DURATION:{receipt_reader.seg_dur:<5} CREATED:{receipt_reader.created_time:<10}"

            logger.info(out_str)


def nmi_scores(latent_space, y_true, n_clusters: int = 10):
    """
    With the provided latent space and true y values, the latent space is clustered using:

    - kmeans
    - GMM
    - DBSCAN

    The NMI score for each clustering output is computed.

    :param latent_space: latent space to cluster
    :param y_true: true labels
    :param n_clusters: number of clusters
    :return: kmeans_nmi, gm_nmi, db_scan_nmi
    """

    # run kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(latent_space)
    kmeans_nmi = normalized_mutual_info_score(y_true, y_pred)

    # run dbscan
    db_scan = DBSCAN()
    y_pred = db_scan.fit_predict(latent_space)
    db_scan_nmi = normalized_mutual_info_score(y_true, y_pred)

    # gaussian model
    gm = GaussianMixture(n_components=10)
    y_pred = gm.fit_predict(latent_space)
    gm_nmi = normalized_mutual_info_score(y_true, y_pred)

    return kmeans_nmi, gm_nmi, db_scan_nmi


def plot_3D(latent_space, y_true, path, logger, genre_filter, loader):
    unique_labels = np.unique(y_true)
    str_labels = loader.decode_label(unique_labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], c=y_true, cmap='tab10', alpha=0.7, s=10)
    colour_bar = plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    colour_bar.set_ticks(unique_labels)
    colour_bar.set_ticklabels(str_labels)

    ax.set_title(f"3D Visualisation of Latent Space (genres: {genre_filter})")
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_zlabel("Axis 3")

    ax.grid(False)
    plt.show()
    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_2D(latent_space, y_true, path, logger, genre_filter, loader):
    unique_labels = np.unique(y_true)
    str_labels = loader.decode_label(unique_labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(latent_space[:, 0], latent_space[:, 1], c=y_true, cmap='tab10', alpha=0.7, s=10)
    colour_bar = plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    colour_bar.set_ticks(unique_labels)
    colour_bar.set_ticklabels(str_labels)

    ax.set_title(f"2D Visualisation of Latent Space (genres: {genre_filter})")
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.grid(False)

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def cluster_statistics(y_true, y_pred, loader):
    cluster_stats = {}

    # convert the encoded labels back to strings
    y_true = loader.decode_label(y_true)
    for i in range(0, len(y_pred)):
        if y_pred[i] not in cluster_stats:
            cluster_stats.update({y_pred[i]: {}})

        if y_true[i] not in cluster_stats[y_pred[i]]:
            cluster_stats[y_pred[i]].update({y_true[i]: 0})

        cluster_stats[y_pred[i]][y_true[i]] += 1

    return cluster_stats

def plot_cluster_statistics(cluster_stats: dict, path, logger):
    nrows = int(np.floor(np.sqrt(len(cluster_stats))))
    ncols = int(np.ceil(len(cluster_stats) / nrows))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 10), layout="constrained")

    row = 0
    col = 0
    for cluster_key, values in cluster_stats.items():
        labels = []
        sizes = []
        for key, value in values.items():
            labels.append(key)
            sizes.append(cluster_stats[cluster_key][key])

        axs[row, col].set_title(f"Genres in Cluster {cluster_key}")
        axs[row, col].pie(sizes, labels=labels, autopct='%1.1f%%')

        row += 1
        if row == nrows:
            row = 0
            col += 1

    # hide the last default plot
    if col < ncols:
        axs[row, col].axis('off')

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_2d_kmeans_boundaries(latent_space, kmeans, logger, path, genre_filter, y_true, h: float = 0.02):
    # colour map
    cmap = ListedColormap(plt.cm.get_cmap("tab20", kmeans.n_clusters).colors)

    # plot the decision boundary
    x_min, x_max = latent_space[:, 0].min() - 1, latent_space[:, 0].max() + 1
    y_min, y_max = latent_space[:, 1].min() - 1, latent_space[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # obtain labels for each point in mesh. Use last trained model.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(grid_points)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    img = plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=cmap, aspect="auto", origin="lower")
    plt.plot(latent_space[:, 0], latent_space[:, 1], "k.", markersize=2)

    # plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
    plt.title(f"K-means Clustering Boundaries \n (genres: {genre_filter})")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # adding the colour bar
    colorbar = plt.colorbar(img, ticks=range(kmeans.n_clusters))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(kmeans.n_clusters)])

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_eigenvalues(path, pca_model: PCA, logger):
    plt.plot([i for i in range(1, pca_model.n_components + 1)], pca_model.explained_variance_, marker="o", linestyle="-", label="Eigenvalues")
    plt.xlabel("Number of Components")
    plt.ylabel("Eigenvalues (log)")
    plt.yscale("log")
    plt.title("PCA Eigenvalues (log scale)")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved plot '{path}'")

def analyse_kmeans(latent_space, logger, max_clusters=20, n_genres=10):
    inertia = []
    k_values = range(1, max_clusters + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(latent_space)
        inertia.append(kmeans.inertia_)

    kneed_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
    elbow_k = kneed_locator.knee
    elbow_inertia = inertia[elbow_k - 1]

    plt.figure(figsize=(8, 6))
    plt.axvline(x=elbow_k, color="red", linestyle="--", label=f"Elbow: k={elbow_k}")
    plt.axhline(y=elbow_inertia, color="green", linestyle="--", label=f"Inertia: {elbow_inertia:.2e}")
    plt.plot(k_values, inertia, color="blue", marker="o", linestyle="-")
    plt.plot(n_genres, inertia[n_genres-1], color="red", marker="o", label=f"Total Genres: {n_genres}")

    plt.xticks(k_values)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Score")
    plt.title("KMeans Inertia")
    plt.legend()
    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def analyse_latent_dims(dim_reducer: PCA | umap.UMAP, loader, logger, path, title, n_clusters=10, max_components=100):
    data = []
    y_true = []

    kmeans_nmi_metrics = []
    gm_nmi_metrics = []
    db_scan_nmi_metrics = []

    best_kmeans = 0
    best_gm = 0
    best_db = 0

    best_kmeans_comp = 0
    best_gm_comp = 0
    best_db_comp = 0

    x_components = [n for n in range(2, max_components + 1)]
    batch_loader = loader.load(split_type="all", normalise=True)

    for x, y in batch_loader:
        x = x.numpy()
        flattened = [i.flatten() for i in x]
        data.extend(flattened)
        y_true.extend(y)

    for n in tqdm(range(2, max_components + 1), desc="Computing latent dimensions"):
        dim_reducer.n_components = n
        latent_space = dim_reducer.fit_transform(data)
        kmeans_nmi, gm_nmi, db_scan_nmi = nmi_scores(latent_space, y_true, n_clusters)
        if kmeans_nmi > best_kmeans:
            best_kmeans = kmeans_nmi
            best_kmeans_comp = n

        if gm_nmi > best_gm:
            best_gm = gm_nmi
            best_gm_comp = n

        if db_scan_nmi > best_db:
            best_db = db_scan_nmi
            best_db_comp = n

        kmeans_nmi_metrics.append(kmeans_nmi)
        gm_nmi_metrics.append(gm_nmi)
        db_scan_nmi_metrics.append(db_scan_nmi)

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Normalised Mutual Information Score")
    plt.plot(x_components, kmeans_nmi_metrics, color="blue", label="Kmeans NMI Scores", alpha=0.7)
    plt.plot(x_components, gm_nmi_metrics, color="red", label="GMM NMI Scores", alpha=0.7)
    plt.plot(x_components, db_scan_nmi_metrics, color="green", label="DB-SCAN NMI Scores", alpha=0.7)

    plt.plot(best_kmeans_comp, best_kmeans, "o", color="pink",label=f"Best KMeans Latent Size: '{best_kmeans_comp}' - {round(best_kmeans, 3)}")
    plt.plot(best_gm_comp, best_gm, "o", color="pink",label=f"Best GMM Latent Size: '{best_gm_comp}' - {round(best_gm, 3)}")
    plt.plot(best_db_comp, best_db, "o", color="pink",label=f"Best DB-SCAN Latent Size: '{best_db_comp}' - {round(best_db, 3)}")

    plt.legend()
    plt.savefig(path)
    plt.close()

    logger.info(f"Saved plot '{path}'")

def load_flatten(batch_loader):
    data = []
    y_true = []
    for x, y in batch_loader:
        x = x.numpy()
        flattened = [i.flatten() for i in x]
        data.extend(flattened)
        y_true.extend(y)

    return data, y_true

def get_dim_model(model_type):
    seed=42
    if model_type.lower() == "pca":
        return PCA(n_components=2, random_state=seed)
    elif model_type == "umap":
        return umap.UMAP(n_components=2, n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=seed)

def get_genre_filter(genres_arg):
    if genres_arg != "all":
        genre_filter = genres_arg.replace(" ", "").split(",")
        n_genres = len(genre_filter)
    else:
        genre_filter = []
        n_genres = 10

    return n_genres, genre_filter

if __name__ == "__main__":
    args = parser.parse_args()

    metrics = {}
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    experiments_dir_root = os.path.join(config.OUTPUT_PATH, "experiments")
    if not os.path.exists(experiments_dir_root):
        os.mkdir(experiments_dir_root)

    if args.info:
        show_info(logger, config)

    if args.uuid:
        uuid_dir = os.path.join(config.OUTPUT_PATH, args.uuid)
        with utils.ReceiptReader(filename=os.path.join(uuid_dir, "receipt.json")) as receipt_reader:
            signal_processor = receipt_reader.signal_processor

        if args.nmi:
            experiments_dir = os.path.join(experiments_dir_root, "nmi-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            # analyses the NMI scores against the latent space size after pca or umap
            # this plots these results on a graph
            # run this with different preprocessed datasets

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            dim_model = get_dim_model(args.nmi)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.nmi.lower()}_latent_dims_analysis.pdf")
            logger.info(f"Latent space analysis for {args.nmi.lower()}")
            analyse_latent_dims(dim_reducer=dim_model, loader=loader, logger=logger, path=path, max_components=200, n_clusters=10, title=f"{signal_processor} Latent Space Analysis for {args.nmi.lower()}")

        if args.eigen:
            experiments_dir = os.path.join(experiments_dir_root, "eigenvalue-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            # plots an eigenvalue plot obtained after performing pca with n_components
            # takes a flag for the number of components pca should use
            # run this with different preprocessed datasets
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            pca_model = PCA(n_components=args.eigen)

            batch_loader = loader.load(split_type="all", normalise=True)
            data, y_true = load_flatten(batch_loader)
            pca_model.fit(data)
            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_eigenvalues.pdf")
            plot_eigenvalues(path=path, pca_model=pca_model, logger=logger)

        if args.boundaries:
            # use pca or umap
            # uses kmeans to produce kmeans boundaries plot

            experiments_dir = os.path.join(experiments_dir_root, "kmeans-boundary-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = load_flatten(batch_loader)
            dim_model = get_dim_model(args.boundaries)

            latent = dim_model.fit_transform(data).astype(np.float64)
            kmeans = KMeans(n_clusters=args.n_clusters)
            y_pred = kmeans.fit_predict(latent)

            # plot boundaries
            path1 = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.boundaries.lower()}_{args.genres}_kmeans_boundaries_{args.n_clusters}.pdf")
            plot_2d_kmeans_boundaries(kmeans=kmeans, latent_space=latent, y_true=y_true, logger=logger, path=path1, genre_filter=args.genres)

            # plot genre spread
            stats = cluster_statistics(y_true=np.array(y_true), y_pred=np.array(y_pred), loader=loader)
            path2 = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.boundaries.lower()}_{args.genres}_genre_spread_{args.n_clusters}.pdf")
            plot_cluster_statistics(cluster_stats=stats, path=path2, logger=logger)

        if args.inertia:
            experiments_dir = os.path.join(experiments_dir_root, "kmeans-inertia-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            n_genres, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)
            data, y_true = load_flatten(batch_loader)
            dim_model = get_dim_model(args.inertia)

            latent_space = dim_model.fit_transform(data)
            max_clusters = args.n_clusters
            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.inertia.lower()}_{args.genres}_kmeans_inertia.pdf")
            analyse_kmeans(latent_space=latent_space, max_clusters=max_clusters, logger=logger, n_genres=n_genres)

        if args.visualise2d:
            # use pca or umap
            # plot latent space

            experiments_dir = os.path.join(experiments_dir_root, "2d-visualise-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = load_flatten(batch_loader)
            dim_model = get_dim_model(args.visualise2d)
            latent = dim_model.fit_transform(data)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.visualise2d.lower()}_{args.genres}_visualisation_2D.pdf")
            plot_2D(latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres, loader=loader)

        if args.visualise3d:
            # use pca or umap
            # plot latent space

            experiments_dir = os.path.join(experiments_dir_root, "3d-visualise-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = load_flatten(batch_loader)
            dim_model = get_dim_model(args.visualise3d)

            dim_model.n_components = 3
            latent = dim_model.fit_transform(data)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.visualise3d.lower()}_{args.genres}_visualisation_3D.pdf")
            plot_3D(latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres, loader=loader)
