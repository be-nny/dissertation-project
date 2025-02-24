import argparse
import os
import random

import matplotlib
import numpy as np
from tqdm import tqdm

import config
import logger
import model

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, homogeneity_score, silhouette_score, calinski_harabasz_score
from model import utils, models
from plot_lib import plotter, interactive_plotter
from preprocessor import preprocessor as p, signal_processor as sp

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - MODEL', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-f", "--fit_new_song", help="Fit a new song")
parser.add_argument("-p", "--path", action="store_true", help="Plots the shortest path between two random starting points")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")
parser.add_argument("-n", "--n_clusters", type=int, help="The number of clusters to find")

def get_genre_filter(genres_arg: str) -> tuple[int, list]:
    """
    Formats a genre filter string

    :param genres_arg: genre filter from CLI arg
    :return: number of genres, formatted genre list
    """

    if genres_arg != "all":
        genre_filter = genres_arg.replace(" ", "").split(",")
        n_genres = len(genre_filter)
    else:
        genre_filter = []
        n_genres = 10

    return n_genres, genre_filter

def show_info(logger: logger.logging.Logger, config: config.Config) -> None:
    """
    Shows all available datasets to in the output directory in 'config.yml'

    :param logger: logger
    :param config: config file
    """

    datasets = os.listdir(config.OUTPUT_PATH)

    for uuid in datasets:
        if uuid[0] != "." and uuid != "experiments":
            path = os.path.join(config.OUTPUT_PATH, uuid)
            receipt_file = os.path.join(path, "receipt.json")
            with utils.ReceiptReader(filename=receipt_file) as receipt_reader:
                out_str = f"{uuid} - {receipt_reader.signal_processor:<15} SAMPLE SIZE: {receipt_reader.total_samples:<5} SEGMENT DURATION:{receipt_reader.seg_dur:<5} CREATED:{receipt_reader.created_time:<10}"

            logger.info(out_str)

def cluster_statistics(y_true: np.ndarray, y_pred: np.ndarray, loader: model.utils.Loader) -> dict:
    """
    Creates a dictionary containing which clusters have what genre in them. Each genre has a count of the number of samples in that cluster with that genre tag

    :param y_true: true label values
    :param y_pred: predicted label values
    :param loader: dataset loader
    :return: the cluster statistics
    """

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

def fit_new(new_file_path: str, model: models.GMMLearner, signal_func_name: str, segment_duration: int, fig: plt.Figure, ax: plt.Axes):
    file_name = os.path.basename(new_file_path).strip().replace("_", " ")

    signal_func = sp.get_type(signal_func_name)
    segment_data = p.apply_signal(path=new_file_path, signal_func=signal_func, segment_duration=segment_duration, sample_rate=22050)

    # if segment duration is 30 seconds and overlap_ratio is set to 0.1, then 27 seconds of the previous window
    # will be included in the next window
    overlap_ratio = 0.1
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

    return fig, ax

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)

    with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
        signal_processor = receipt.signal_processor
        segment_duration = receipt.seg_dur

    folder = f"{signal_processor}_{args.uuid}_{args.genres}_{args.n_clusters}"
    root = f"{config.OUTPUT_PATH}/gaussian_model/{folder}"
    if not os.path.exists(root):
        os.makedirs(root)

    logger.info("Running Gaussian Model")

    # create a dataloader
    _, genre_filter = get_genre_filter(args.genres)
    loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
    batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=True)

    # create metric learner (gaussian mixture model)
    gmm_model = models.GMMLearner(loader=batch_loader, n_clusters=args.n_clusters)
    gmm_model.create_latent()
    latent_space, y_pred, y_true = gmm_model.get_latent(), gmm_model.get_y_pred(), gmm_model.get_y_true()
    covar = gmm_model.gaussian_model.covariances_[0]
    inv_covar = np.linalg.inv(covar)

    # correlation
    n_neighbours_total = [1, 5, 10, 50]
    tqdm_loop = tqdm(n_neighbours_total, desc="Computing correlation matrices", unit="iter")
    for n_neighbours in tqdm_loop:
        t_corr, p_corr = utils.correlation(latent_space=latent_space, y_true=y_true, covar=inv_covar, n_neighbours=n_neighbours)
        cf_matrix = confusion_matrix(loader.decode_label(t_corr), loader.decode_label(p_corr))
        class_labels = sorted(set(loader.decode_label(t_corr)) | set(loader.decode_label(p_corr)))
        path = f"{root}/{n_neighbours}_nearest_neighbours_confusion_mat.pdf"
        plotter.plot_correlation(cf_matrix=cf_matrix, class_labels=class_labels, n_neighbours=n_neighbours, path=path)

    # plot correlation accuracy
    path = f"{root}/correlation_accuracy.pdf"
    plotter.plot_correlation_accuracy(latent_space=latent_space, y_true=y_true, covariance_mat=inv_covar, path=path)
    logger.info(f"Saved plot '{path}'")

    # get cluster stats for tree maps
    cluster_stats = cluster_statistics(y_true=y_true, y_pred=y_pred, loader=loader)

    # find the largest genre per cluster
    prominent_cluster_genre = {}
    for cluster_key, values in cluster_stats.items():
        sort_by_value = dict(sorted(values.items(), key=lambda item: item[1]))
        largest_genre = list(sort_by_value)[-1]
        prominent_cluster_genre.update({cluster_key: largest_genre})
    prominent_cluster_genre = dict(sorted(prominent_cluster_genre.items(), key=lambda item: item[0]))

    # plot treemaps
    path = f"{root}/tree_map.pdf"
    plotter.plot_cluster_statistics(cluster_stats=cluster_stats, path=path)
    logger.info(f"Saved plot '{path}'")

    # show window
    data_points = utils.create_custom_points(latent_space=latent_space, y_pred=y_pred, y_true=y_true, raw_paths=loader.get_associated_paths(), covar=inv_covar)
    path = f"{root}/gaussian_plot.pdf"
    title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
    ax, fig = interactive_plotter.interactive_gmm(gmm=gmm_model.gaussian_model, data_points=data_points, title=title, path=path)
    logger.info(f"Saved plot '{path}'")

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

    if args.path and len(shortest_path) > 1:
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
        path = f"{root}/gaussian_plot_shortest_path.pdf"
        plt.savefig(path)
        logger.info(f"Saved plot '{path}'")

    # fit new song to plot the 'song evolution'
    if args.fit_new_song:
        ax, fig = fit_new(new_file_path=args.fit_new_song, model=gmm_model, signal_func_name=signal_processor, segment_duration=segment_duration, fig=fig, ax=ax)
        file_name = os.path.basename(args.fit_new_song).strip().replace("_", " ")
        path = f"{root}/gaussian_plot_with_{file_name}.pdf"
        plt.savefig(path)
        logger.info(f"Saved plot '{path}'")

    plt.autoscale()
    logger.info("Displaying Window")

    prom_genres = "Most Common Genre per Cluster: \n" + '\n'.join([f"> Cluster {k}: {v}" for k, v in prominent_cluster_genre.items()])
    logger.info(prom_genres)
    logger.info(f"Homogeneity Score: {homogeneity_score(y_true, y_pred)}")
    logger.info(f"Calinski Harabasz Score: {calinski_harabasz_score(latent_space, y_pred)}")
    plt.show()
