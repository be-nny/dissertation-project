import argparse
import os.path
import matplotlib
import numpy as np
import logger
import model

from tqdm import tqdm
from utils import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, homogeneity_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score
from model import utils, models
from plot_lib import plotter, interactive_plotter
from preprocessor import preprocessor as p, signal_processor as sp
from scipy.spatial.distance import mahalanobis

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(prog='Music Genre Analysis Tool - Clustering', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use, or a list of comma seperated uuid's")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-t", "--cluster_type", choices=["kmeans", "gmm"], help="Model type to use (kmeans, gmm)")
parser.add_argument("-f", "--fit_new_song", help="Fit a new song")
parser.add_argument("-p", "--path", action="store_true", help="Plots the shortest path between two random starting points")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")
parser.add_argument("-n", "--n_clusters", type=int, help="The number of clusters to find")

def _prominent_genres(cluster_stats: dict):
    """
    Find the most prominent genre in each cluster

    :param cluster_stats: cluster statistics
    :return: dict of most prominent genres as 'cluster_n':'genre'
    """

    prominent_cluster_genre = {}
    for cluster_key, values in cluster_stats.items():
        sort_by_value = dict(sorted(values.items(), key=lambda item: item[1]))
        largest_genre = list(sort_by_value)[-1]
        prominent_cluster_genre.update({cluster_key: largest_genre})
    prominent_cluster_genre = dict(sorted(prominent_cluster_genre.items(), key=lambda item: item[0]))
    return prominent_cluster_genre

def _add_shortest_path(shortest_path, data_points, start_point, end_point, path, ax):
    """
    Add the shortest between two points in the latent space

    :param shortest_path: shortest path between the points
    :param data_points: custom data ponts
    :param start_point: start point
    :param end_point: end point
    :param path: path to save
    :param ax: figure to plot to
    """

    padding = 3
    ax.plot(shortest_path[:, 0], shortest_path[:, 1], color="pink", label="Shortest path")

    # annotate which song is which point
    for p1 in shortest_path:
        for p2 in data_points:
            if np.all(p1 == p2.point):
                name = os.path.basename(p2.raw_path)
                ax.annotate(name, (p1[0], p1[1]), textcoords="offset points", xytext=(5, 2), ha='center', fontsize=5, alpha=0.5)
                break

    # adjust view
    if start_point[0] < end_point[0]:
        plt.xlim(start_point[0] - padding, end_point[0] + padding)
    else:
        plt.xlim(end_point[0] - padding, start_point[0] + padding)

    if start_point[1] < end_point[1]:
        plt.ylim(start_point[1] - padding, end_point[1] + padding)
    else:
        plt.ylim(end_point[1] - padding, start_point[1] + padding)

    # save
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.autoscale()

def _fit_new(new_file_path: str, model: models.MetricLeaner, signal_func_name: str, segment_duration: int, sample_rate: int, fig: plt.Figure, ax: plt.Axes, path: str, overlap_ratio: float = 0.3):
    """
    Fit a new song to show its evolution. If segment duration is 30 seconds and overlap_ratio is set to 0.1, then 27 seconds of the previous window will be included in the next window

    :param new_file_path: path to new song
    :param model: the clustering model
    :param signal_func_name: signal function to apply
    :param segment_duration: segment duration
    :param sample_rate: sample rate
    :param fig: figure
    :param ax: axis
    :param overlap_ratio: overlap ratio
    :param path: path to save
    """

    file_name = os.path.basename(new_file_path).strip().replace("_", " ")

    signal_func = sp.get_type(signal_func_name)
    segment_data = p.apply_signal(path=new_file_path, signal_func=signal_func, segment_duration=segment_duration, sample_rate=sample_rate)

    # if segment duration is 30 seconds and overlap_ratio is set to 0.1, then 27 seconds of the previous window
    # will be included in the next window
    signal_func_width = segment_data.shape[-1]

    stride = int(signal_func_width * overlap_ratio)
    window_size = int(signal_func_width)

    merged_signals = np.concatenate(segment_data, axis=-1)
    out_shape = ((merged_signals.shape[1] - window_size) // stride + 1, merged_signals.shape[0], window_size)
    strides = (merged_signals.strides[1] * stride, merged_signals.strides[0], merged_signals.strides[1])
    overlapping_reg = np.lib.stride_tricks.as_strided(merged_signals, shape=out_shape, strides=strides)

    flattened_overlapping_reg = [arr.flatten() for arr in overlapping_reg]
    new_fitted, _ = model.fit_new(flattened_overlapping_reg)
    ax.scatter(new_fitted[:, 0], new_fitted[:, 1], color="purple", s=10)
    ax.scatter(new_fitted[0][0], new_fitted[0][1], color="blue", marker="^", s=10, label="start")
    ax.scatter(new_fitted[-1][0], new_fitted[-1][1], color="red", marker="s", s=10, label="end")
    ax.plot(new_fitted[:, 0], new_fitted[:, 1], color="purple", linestyle="-", label=file_name, linewidth=1)

    ax.legend()
    plt.savefig(path, bbox_inches='tight')

    return fig, ax


def _mahalanobis_distance_matrix(X, VI):
    """
    Compute pairwise Mahalanobis distance matrix for dataset X given inverse covariance matrix VI.

    :param X: latent space
    :param VI: inverse covariance matrix
    :return: distance matrix
    """
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = mahalanobis(X[i], X[j], VI)

    return dist_matrix

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)
    elif not args.cluster_type:
        parser.error("missing flags. '-t,--type' must be set")

    # runs the specified model (either 'kmeans' or 'gmm')
    # saves any relevant figures and displays the window
    with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
        signal_processor = receipt.signal_processor
        segment_duration = receipt.seg_dur

    folder = f"{signal_processor}_{args.uuid}_{args.genres}_{args.n_clusters}"
    root = f"{config.OUTPUT_PATH}/_{args.cluster_type}/{folder}"
    if not os.path.exists(root):
        os.makedirs(root)

    # create a dataloader
    _, genre_filter = get_genre_filter(args.genres)
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
    loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=True)

    # create metric learner
    metric_leaner = models.MetricLeaner(loader=loader, n_clusters=args.n_clusters, cluster_type=args.cluster_type)
    latent_space, y_pred, y_true = metric_leaner.get_latent(), metric_leaner.get_y_pred(), metric_leaner.get_y_true()

    # getting covariance matrix (if required)
    inv_covar = None
    if args.cluster_type == "gmm":
        logger.info(f"Using '{args.cluster_type}' - set distance metric to 'mahalanobis'")
        covar = metric_leaner.cluster_model.covariances_[0]
        inv_covar = np.linalg.inv(covar)
    else:
        logger.info(f"Using '{args.cluster_type}' - set distance metric to 'euclidean'")

    # get cluster stats for tree maps
    cluster_stats = utils.cluster_statistics(y_true=y_true, y_pred=y_pred, loader=loader)

    # create a set of data points that can be used for interactive plot
    data_points = utils.create_custom_points(latent_space=latent_space, y_pred=y_pred, y_true=y_true, raw_paths=loader.get_associated_paths(), covar=inv_covar)

    # create graph for shortest path
    graph = utils.connected_graph(latent_space, inv_covar)
    start_point = latent_space[np.random.randint(len(latent_space))]
    end_point = latent_space[np.random.randint(len(latent_space))]

    s = ','.join(str(p) for p in start_point)
    e = ','.join(str(p) for p in end_point)
    _, shortest_path = utils.shortest_path(graph, s, e)

    # plotting the shortest path between two random points
    if not len(shortest_path):
        logger.warning("No shortest path could be found. Try adjusting nearset neighbours for `utils.connected_graph`")

    # correlation
    n_neighbours_total = [1, 5, 10, 50]
    tqdm_loop = tqdm(n_neighbours_total, desc="Computing correlation matrices", unit="iter")
    for n_neighbours in tqdm_loop:
        t_corr, p_corr = utils.correlation(latent_space=latent_space, y_true=y_true, covar=inv_covar, n_neighbours=n_neighbours)
        f1, precision, recall, acc = utils.correlation_metrics(t_corr, p_corr)

        cf_matrix = confusion_matrix(loader.decode_label(t_corr), loader.decode_label(p_corr))
        class_labels = sorted(set(loader.decode_label(t_corr)) | set(loader.decode_label(p_corr)))
        path = f"{root}/{n_neighbours}_nearest_neighbours_confusion_mat.pdf"
        plotter.plot_correlation_conf_mat(cf_matrix=cf_matrix, class_labels=class_labels, n_neighbours=n_neighbours, path=path, f1_score=f1, recall=recall, precision=precision, accuracy=acc)

    # plot tree maps
    path = f"{root}/tree_map.pdf"
    plotter.plot_tree_map(cluster_stats=cluster_stats, path=path)
    logger.info(f"Saved plot '{path}'")

    # show window
    path = f"{root}/{args.cluster_type}_plot.pdf"
    title = f"{str(args.cluster_type).upper()} cluster boundaries with {signal_processor} applied"
    if args.cluster_type == "gmm":
        ax, fig = interactive_plotter.interactive_gmm(gmm=metric_leaner.cluster_model, data_points=data_points, title=title, path=path)
        logger.info(f"Saved plot '{path}'")
    elif args.cluster_type == "kmeans":
        ax, fig = interactive_plotter.interactive_kmeans(kmeans=metric_leaner.cluster_model, data_points=data_points, title=title, path=path)
        logger.info(f"Saved plot '{path}'")

    # show the shortest path
    if args.path and len(shortest_path) > 1:
        path = f"{root}/{args.cluster_type}_plot_shortest_path.pdf"
        _add_shortest_path(shortest_path=shortest_path, data_points=data_points, start_point=start_point, end_point=end_point, ax=ax, path=path)
        logger.info(f"Saved plot '{path}'")

    # fit new song to plot the 'song evolution'
    if args.fit_new_song:
        file_name = os.path.basename(args.fit_new_song).strip().replace("_", " ")
        path = f"{root}/gmm_with_{file_name}.pdf"
        _fit_new(new_file_path=args.fit_new_song, model=metric_leaner, signal_func_name=signal_processor, sample_rate=config.SAMPLE_RATE, segment_duration=segment_duration, fig=fig, ax=ax, path=path)
        logger.info(f"Saved plot '{path}'")

    logger.info("Displaying Window")

    prom_genres = "Most Common Genre per Cluster: " + ', '.join([f"{k}: {v}" for k, v in _prominent_genres(cluster_stats).items()])
    logger.info(prom_genres)
    logger.info(f"homogeneity score: {homogeneity_score(y_true, y_pred):.4f}")
    logger.info(f"davies bouldin score: {davies_bouldin_score(latent_space, y_pred):.4f}")
    logger.info(f"calinski harabasz score: {calinski_harabasz_score(latent_space, y_pred):.4f}")

    if inv_covar is not None:
        dist_matrix = _mahalanobis_distance_matrix(latent_space, inv_covar)
        logger.info(f"silhouette score: {silhouette_score(dist_matrix, y_pred, metric='precomputed'):.4f}")
    else:
        logger.info(f"silhouette score: {silhouette_score(latent_space, y_pred):.4f}")

    plt.show()

