import argparse
import os

import librosa
import matplotlib
import numpy as np

import config
import logger
import model

from matplotlib import pyplot as plt

from model import utils, models
from plot_lib import plotter, interactive_plotter
from preprocessor import preprocessor as p, signal_processor as sp

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - MODEL', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-r", "--run", help="Runs the model")
parser.add_argument("-f", "--fit_new_song", help="Fit a new song. '-r' must be called first")
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

def fit_new(path: str, model: models.GMMLearner, signal_func_name: str, segment_duration: int, fig: plt.Figure, ax: plt.Axes):
    file_name = os.path.basename(path).strip().replace("_", " ")

    signal_func = sp.get_type(signal_func_name)
    segment_data = p.apply_signal(path=path, signal_func=signal_func, segment_duration=segment_duration)

    stride = int(segment_data.shape[-1] * (15/30))
    window_size = int(segment_data.shape[-1] * 2)

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

    if args.run == "agglomerative":
        with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
            signal_processor = receipt.signal_processor

        root = f"{config.OUTPUT_PATH}/models/metric_learning/{args.uuid}"
        if not os.path.exists(root):
            os.makedirs(root)

        logger.info("Running Model: Agglomerative Clustering")

        # create a dataloader
        _, genre_filter = get_genre_filter(args.genres)
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=True)

        # create metric learner
        agg = models.HierarchicalClustering(loader=batch_loader)
        agg.create_latent()

        # plot dendrogram
        plt.title("Agglomerative Clustering Dendrogram")
        raw_labels = loader.get_associated_paths()
        plotter.plot_dendrogram(agg.agglomerative, labels=raw_labels, truncate_mode="level")
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    if args.run == "metric_learning":
        with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
            signal_processor = receipt.signal_processor
            segment_duration = receipt.seg_dur

        root = f"{config.OUTPUT_PATH}/models/metric_learning/{args.uuid}"
        if not os.path.exists(root):
            os.makedirs(root)

        logger.info("Running Model: Metric Learning Model")

        # create a dataloader
        _, genre_filter = get_genre_filter(args.genres)
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=True)

        # create metric learner
        metric_l = models.GMMLearner(loader=batch_loader, n_clusters=args.n_clusters)
        metric_l.create_latent()
        latent_space, y_pred, y_true = metric_l.get_latent(), metric_l.get_y_pred(), metric_l.get_y_true()

        # get cluster stats for tree maps
        path = f"{root}/tree_map_{args.n_clusters}_{args.genres}.pdf"
        cluster_stats = cluster_statistics(y_true=y_true, y_pred=y_pred, loader=loader)
        plotter.plot_cluster_statistics(cluster_stats=cluster_stats, path=path, logger=logger)

        # show window
        covar = metric_l.gaussian_model.covariances_[0]
        inv_covar = np.linalg.inv(covar)
        data_points = utils.create_custom_points(latent_space=latent_space, y_pred=y_pred, y_true=y_true, raw_paths=loader.get_associated_paths(), covar=inv_covar)
        path = f"{root}/gaussian_plot_{args.n_clusters}_{args.genres}.pdf"
        title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
        ax, fig = interactive_plotter.interactive_gmm(gmm=metric_l.gaussian_model, data_points=data_points, title=title, path=path)

        if args.fit_new_song:
            ax, fig = fit_new(path=args.fit_new_song, model=metric_l, signal_func_name=signal_processor, segment_duration=segment_duration, fig=fig, ax=ax)

        plt.show()

    if args.run == "genre_classifier":
        with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
            signal_processor = receipt.signal_processor

        root = f"{config.OUTPUT_PATH}/models/multi_class/{args.uuid}"
        if not os.path.exists(root):
            os.makedirs(root)

        logger.info("Running Model: Multi-class Model")

        # create a dataloader
        _, genre_filter = get_genre_filter(args.genres)
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=16)
        batch_loader = loader.load(split_type="train", normalise=True, genre_filter=genre_filter)
        input_dims = int(loader.input_shape[0])
        multi_classifier = models.GenreClassifier(n_classes=10)
        models.train_genre_classifier(model=multi_classifier, loader=batch_loader, n_epochs=500)