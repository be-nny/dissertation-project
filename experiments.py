import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import os.path
import numpy as np
import umap.umap_ as umap
import matplotlib
import config
import logger

from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from model import utils

matplotlib.use('TkAgg')

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - EXPERIMENTS', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-n", "--nmi", type=str, help="Plots NMI score against the number of UMAP or PCA components. Set flag to 'pca' or 'umap'")
parser.add_argument("-e", "--eigen", type=int, help="Plots the eigenvalues obtained after performing PCA. Takes value for max n_components")
parser.add_argument("-b", "--boundaries", help="Plots 2D Kmeans Boundaires of UMAP or PCA. '-nc' must be set for the number of clusters. '-g' must also be set.")
parser.add_argument("-t", "--inertia", help="Plots number of clusters against kmeans inertia score for UMAP or PCA. '-g' must also be set. '-nc' must be set for the max. number of clusters")
parser.add_argument("-v2", "--visualise2d", help="Plots 2D Latent Space of UMAP or PCA. '-g' must also be set.")
parser.add_argument("-v3", "--visualise3d", help="Plots 3D Latent Space of UMAP or PCA. '-g' must also be set.")
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

def plot_2d_kmeans_boundaries(latent_space, kmeans, logger, path, genre_filter, y_true, h: float = 0.02):
    # plot the decision boundary
    x_min, x_max = latent_space[:, 0].min() - 1, latent_space[:, 0].max() + 1
    y_min, y_max = latent_space[:, 1].min() - 1, latent_space[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect="auto", origin="lower")
    plt.plot(latent_space[:, 0], latent_space[:, 1], "k.", markersize=2)

    # plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
    plt.title(f"K-means Clustering Boundaries (genres: {genre_filter})")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_eigenvalues(path, pca_model: PCA, logger):
    plt.plot([i for i in range(1, pca_model.n_components + 1)], pca_model.explained_variance_, marker="o", linestyle="-",label="Eigenvalues")
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

    plt.figure(figsize=(8, 8))
    plt.plot(k_values, inertia, color="blue", marker="o", linestyle="-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Score")
    plt.title("KMeans Inertia against Number of Clusters")
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

    plt.plot(best_kmeans_comp, best_kmeans, "o", color="pink", label=f"Best KMeans Latent Size: '{best_kmeans_comp}' - {round(best_kmeans, 3)}")
    plt.plot(best_gm_comp, best_gm, "o", color="pink", label=f"Best GMM Latent Size: '{best_gm_comp}' - {round(best_gm, 3)}")
    plt.plot(best_db_comp, best_db, "o", color="pink", label=f"Best DB-SCAN Latent Size: '{best_db_comp}' - {round(best_db, 3)}")

    plt.legend()
    plt.savefig(path)
    plt.close()

    logger.info(f"Saved plot '{path}'")

def get_dim_model(model_type):
    if model_type.lower() == "pca":
        return PCA(n_components=2)
    elif model_type == "umap":
        return umap.UMAP(n_components=2, init='random', n_neighbors=10, spread=3, min_dist=0.3, repulsion_strength=2, learning_rate=1.5, n_epochs=500, random_state=42)

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

    experiments_dir = os.path.join(config.OUTPUT_PATH, "experiments")
    if not os.path.exists(experiments_dir):
        os.mkdir(experiments_dir)

    if args.info:
        show_info(logger, config)

    if args.uuid:
        uuid_dir = os.path.join(config.OUTPUT_PATH, args.uuid)
        with utils.ReceiptReader(filename=os.path.join(uuid_dir, "receipt.json")) as receipt_reader:
            signal_processor = receipt_reader.signal_processor

        if args.nmi:
            # analyses the NMI scores against the latent space size after pca or umap
            # this plots these results on a graph
            # run this with different preprocessed datasets

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)

            dim_model = get_dim_model(args.nmi)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.nmi.lower()}_latent_dims_analysis.pdf")
            logger.info(f"Latent space analysis for {args.nmi.lower()}")
            analyse_latent_dims(dim_reducer=dim_model, loader=loader, logger=logger, path=path, max_components=200, n_clusters=10, title=f"{signal_processor} Latent Space Analysis for {args.nmi.lower()}")

        if args.eigen:
            # plots an eigenvalue plot obtained after performing pca with n_components
            # takes a flag for the number of components pca should use
            # run this with different preprocessed datasets

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)

            pca_model = PCA(n_components=args.eigen)
            data = []
            batch_loader = loader.load(split_type="all", normalise=True)
            print(loader.input_shape)
            for x, y in batch_loader:
                x = x.numpy()
                flattened = [i.flatten() for i in x]
                data.extend(flattened)

            pca_model.fit(data)
            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_eigenvalues.pdf")
            plot_eigenvalues(path=path, pca_model=pca_model, logger=logger)

        if args.boundaries:
            # use pca or umap
            # uses kmeans to produce kmeans boundaries plot

            _, genre_filter = get_genre_filter(args.genres)

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data = []
            y_true = []
            for x, y in batch_loader:
                x = x.numpy()
                flattened = [i.flatten() for i in x]
                data.extend(flattened)
                y_true.extend(y)

            dim_model = get_dim_model(args.boundaires)

            latent = dim_model.fit_transform(data)
            kmeans = KMeans(n_clusters=args.n_clusters)
            kmeans.fit(latent)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.boundaires.lower()}_{args.genres}_kmeans_boundaries_{args.n_clusters}.pdf")
            plot_2d_kmeans_boundaries(kmeans=kmeans, latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres)

        if args.inertia:
            data = []
            y_true = []
            n_genres, genre_filter = get_genre_filter(args.genres)

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)
            for x, y in batch_loader:
                x = x.numpy()
                flattened = [i.flatten() for i in x]
                data.extend(flattened)
                y_true.extend(y)

            dim_model = get_dim_model(args.inertia)

            latent_space = dim_model.fit_transform(data)
            max_clusters = args.n_clusters
            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.inertia.lower()}_{args.genres}_kmeans_inertia.pdf")
            analyse_kmeans(latent_space=latent_space, max_clusters=max_clusters, logger=logger, n_genres=n_genres)

        if args.visualise2d:
            # use pca or umap
            # plot latent space
            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data = []
            y_true = []
            for x, y in batch_loader:
                x = x.numpy()
                flattened = [i.flatten() for i in x]
                data.extend(flattened)
                y_true.extend(y)

            dim_model = get_dim_model(args.visualise2d)
            latent = dim_model.fit_transform(data)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.visualise2d.lower()}_{args.genres}_visualisation_2D.pdf")
            plot_2D(latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres, loader=loader)

        if args.visualise3d:
            # use pca or umap
            # plot latent space
            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data = []
            y_true = []
            for x, y in batch_loader:
                x = x.numpy()
                flattened = [i.flatten() for i in x]
                data.extend(flattened)
                y_true.extend(y)

            dim_model = get_dim_model(args.visualise3d)
            dim_model.n_components = 3
            latent = dim_model.fit_transform(data)

            path = os.path.join(experiments_dir, f"{args.uuid}_{signal_processor}_{args.visualise3d.lower()}_{args.genres}_visualisation_3D.pdf")
            plot_3D(latent_space=latent, y_true=y_true, logger=logger, path=path, genre_filter=args.genres, loader=loader)
