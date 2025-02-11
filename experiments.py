import os
import argparse
import os.path
import umap.umap_ as umap
from sklearn.manifold import TSNE

import config
import logger

from plot_lib.plotter import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from model import utils

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - EXPERIMENTS', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-nc", "--n_clusters", type=int, help="number of clusters")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")

parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-e", "--eigen", type=int, help="Plots the eigenvalues obtained after performing PCA. Takes value for max n_components")
parser.add_argument("-b", "--boundaries", help="Plots 2D Kmeans Boundaries of UMAP or PCA. '-nc' must be set for the number of clusters. '-g' must also be set.")
parser.add_argument("-s", "--gaussian", help="Plots 2D Gaussian Boundaries of UMAP or PCA. '-nc' must be set for the number of clusters. '-g' must also be set.")
parser.add_argument("-t", "--inertia", help="Plots number of clusters against kmeans inertia score for UMAP or PCA. '-g' must also be set. '-nc' must be set for the max. number of clusters")
parser.add_argument("-v2", "--visualise2d", help="Plots 2D Latent Space of UMAP or PCA. '-g' must also be set.")
parser.add_argument("-v3", "--visualise3d", help="Plots 3D Latent Space of UMAP or PCA. '-g' must also be set.")

BATCH_SIZE = 512

def show_info(logger: logger.logging.Logger, config: config.Config) -> None:
    """
    Shows all available datasets to in the output directory in 'config.yml'

    :param logger: logger
    :param config: config file
    """
    datasets = os.listdir(config.OUTPUT_PATH)

    for uuid in datasets:
        if uuid[0] != "." and uuid != "experiments" and uuid != "models":
            path = os.path.join(config.OUTPUT_PATH, uuid)
            receipt_file = os.path.join(path, "receipt.json")
            with utils.ReceiptReader(filename=receipt_file) as receipt_reader:
                out_str = f"{uuid} - {receipt_reader.signal_processor:<15} SAMPLE SIZE: {receipt_reader.total_samples:<5} SEGMENT DURATION:{receipt_reader.seg_dur:<5} CREATED:{receipt_reader.created_time:<10}"

            logger.info(out_str)

def load_flatten(batch_loader):
    flattened_data = []
    flattened_y_true = []
    for x, y in batch_loader:
        x = x.numpy()
        flattened = [i.flatten() for i in x]
        flattened_data.extend(flattened)
        flattened_y_true.extend(y)

    return np.array(flattened_data), np.array(flattened_y_true)

def get_dim_model(model_type):
    seed = 42
    if model_type.lower() == "pca":
        return PCA(n_components=2, random_state=seed)
    elif model_type == "umap":
        return umap.UMAP(n_components=2, n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=seed)
    elif model_type == "tsne":
        return TSNE(n_components=2, random_state=seed)
    else:
        raise TypeError("Model type must be 'pca' or 'umap' or 'tsne'")

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

        if args.eigen:
            # plots an eigenvalue plot obtained after performing pca with n_components
            # takes a flag for the number of components pca should use
            # run this with different preprocessed datasets

            experiments_dir = os.path.join(experiments_dir_root, "eigenvalue-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            pca_model = PCA(n_components=args.eigen)

            batch_loader = loader.load(split_type="all", normalise=True)
            data, y_true = load_flatten(batch_loader)
            pca_model.fit(data)
            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_eigenvalues.pdf")

            title = f"PCA Eigenvalues (log scale) with {signal_processor} applied"
            plot_eigenvalues(path=path, pca_model=pca_model, logger=logger, title=title)

        if args.gaussian:
            experiments_dir = os.path.join(experiments_dir_root, "gaussian-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = load_flatten(batch_loader)
            dim_model = get_dim_model(args.gaussian)
            latent_data = dim_model.fit_transform(data).astype(np.float64)

            gmm = GaussianMixture(n_components=args.n_clusters, random_state=42, covariance_type='full')
            y_pred = gmm.fit_predict(latent_data)

            path1 = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.gaussian.lower()}_{args.genres}_gaussian_boundaries_{args.n_clusters}.pdf")
            title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
            plot_gmm(gmm=gmm, X=latent_data, labels=y_pred, path=path1, logger=logger, title=title)

            stats = utils.cluster_statistics(y_true=np.array(y_true), y_pred=np.array(y_pred), loader=loader, logger=logger)
            path2 = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.gaussian.lower()}_{args.genres}_tree_plot_{args.n_clusters}.pdf")
            plot_cluster_statistics(cluster_stats=stats, path=path2, logger=logger)

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
            title = f"K-Means cluster boundaries with {signal_processor} applied"
            plot_2d_kmeans_boundaries(kmeans=kmeans, latent_space=latent, logger=logger, path=path1, genre_filter=args.genres, title=title)

            # plot genre spread
            stats = utils.cluster_statistics(y_true=np.array(y_true), y_pred=np.array(y_pred), loader=loader)
            path2 = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.boundaries.lower()}_{args.genres}_tree_plot_{args.n_clusters}.pdf")
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
            title = f"KMeans Inertia with {signal_processor} applied"
            plot_inertia(latent_space=latent_space, max_clusters=max_clusters, logger=logger, n_genres=n_genres, path=path, title=title)

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
            title = f"Latent Space in 2D with {signal_processor} applied using {args.visualise2d.lower()}"
            plot_2D(latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres, loader=loader, title=title)

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
            title = f"Latent Space in 3D with {signal_processor} applied using {args.visualise2d.lower()}"
            plot_3D(latent_space=latent, y_true=np.array(y_true), logger=logger, path=path, genre_filter=args.genres, loader=loader, title=title)
