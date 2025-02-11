import argparse
import os

import librosa
import matplotlib
import numpy as np

import config
import logger
import model
import preprocessor.signal_processor

from matplotlib import pyplot as plt

from model import utils, models
from plot_lib import plotter, interactive_plotter
from preprocessor import signal_processor, preprocessor

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - MODEL', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-r", "--run", help="Runs the model")
parser.add_argument("-f", "--fit_new", help="Fit a new song. '-r' must be called first")
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

def fit_new(path: str, sr, model: models.MetricLearner, signal_func_name: str, segment_duration: int, fig: plt.Figure, ax: plt.Axes):
    file_name = os.path.basename(path)

    offset = 5
    offset_sr = offset * sr
    segment_duration_sr = segment_duration * sr

    signal_func = preprocessor.signal_processor.get_type(signal_func_name)
    segment_data = preprocessor.preprocessor.apply_signal(path=path, signal_func=signal_func, segment_duration=segment_duration)
    strides = np.lib.stride_tricks.as_strided(segment_data, (len(segment_data) - segment_duration_sr + offset_sr, segment_duration_sr), 2 * segment_data.strides)
    flattened_strides = [segment.flatten() for segment in strides]

    new_fitted = model.fit_new(flattened_strides)

    ax.plot(new_fitted[:, 0], new_fitted[:, 1], 'o', color="red", label=file_name)
    ax.legend()

    return fig, ax

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)

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
        metric_l = models.MetricLearner(loader=batch_loader, n_clusters=args.n_clusters)
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

        if args.fit_new:
            _, sr = librosa.load(path, sr=None)
            ax, fig = fit_new(path=args.fit_new, sr=sr, model=metric_l, signal_func_name=signal_processor, segment_duration=segment_duration, fig=fig, ax=ax)

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
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="train", normalise=True, genre_filter=genre_filter, flatten=True)
        input_dims = int(loader.input_shape[0])
        multi_classifier = models.GenreClassifier(input_dims=input_dims, hidden_dims=[input_dims*2, input_dims*2], output_dims=len(genre_filter))
        models.train_genre_classifier(model=multi_classifier, loader=batch_loader, n_epochs=500)