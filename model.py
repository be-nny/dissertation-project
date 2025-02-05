import argparse
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import config
import logger
import model

from model import utils, models
from preprocessor import preprocessor
from plot_lib import plotter, interactive_plotter
matplotlib.use('TkAgg')


parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - MODEL', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-r", "--run", action="store_true", help="Runs the model")
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

if __name__ == "__main__":
    args = parser.parse_args()
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    if args.info:
        show_info(logger, config)

    if args.run:
        with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{args.uuid}/receipt.json')) as receipt:
            signal_processor = receipt.signal_processor

        root = f"{config.OUTPUT_PATH}/models/metric_learning/{args.uuid}"
        if not os.path.exists(root):
            os.makedirs(root)

        logger.info("Running Model")

        # create a dataloader
        _, genre_filter = get_genre_filter(args.genres)
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=model.BATCH_SIZE)
        batch_loader = loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

        # create metric learner
        metric_l = models.MetricLearner(loader=batch_loader, n_clusters=args.n_clusters)
        metric_l.create_latent()
        latent_space, y_pred, y_true = metric_l.get_latent(), metric_l.get_y_pred(), metric_l.get_y_true()

        data_points = utils.create_custom_points(latent_space=latent_space, y_pred=y_pred, y_true=y_true, raw_paths=loader.get_associated_paths())
        title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
        ax, fig = interactive_plotter.interactive_gmm(gmm=metric_l.gaussian_model, data_points=data_points, title=title)

        plt.show()

        # if args.fit_new:
        #     file_name = os.path.basename(args.fit_new).replace("_", " ")
        #     logger.info(f"Fitting new song: {file_name}")
        #
        #     new_data = preprocessor.apply_signal(path=args.fit_new, segment_duration=15, signal_func=preprocessor.signal_processor.get_type(signal_processor))
        #     new_data = [i.flatten() for i in new_data]
        #     new_latent, new_y_pred = metric_l.fit_new(new_data)
        #
        #     # plot gaussian plot
        #     path = f"{root}/gaussian_plot_{args.n_clusters}_{args.genres}_{file_name}.pdf"
        #     title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
        #     plotter.plot_gmm(gmm=metric_l.gaussian_model, X=latent_space, path=path, labels=y_pred, new_data=new_latent, new_label=file_name, logger=logger, title=title)
        #
        #     # get cluster stats for tree maps
        #     path = f"{root}/tree_map_{args.n_clusters}_{args.genres}_{file_name}.pdf"
        #     title = f"Treemap with {signal_processor} applied"
        #     cluster_stats = cluster_statistics(y_true=y_true, y_pred=y_pred, loader=loader)
        #     plotter.plot_cluster_statistics(cluster_stats=cluster_stats, path=path, logger=logger, title=title)
        #
        #     # getting recommendations
        #     raw_paths = loader.get_associated_paths()
        #     nearest_neighbours = utils.song_recommendation(latent_space=latent_space, raw_paths=raw_paths, points=new_latent, n_neighbours=3)
        #     print(nearest_neighbours)
        # else:
        #     # get cluster stats for tree maps
        #     path = f"{root}/tree_map_{args.n_clusters}_{args.genres}.pdf"
        #     title = f"Treemap with {signal_processor} applied"
        #     cluster_stats = cluster_statistics(y_true=y_true, y_pred=y_pred, loader=loader)
        #     plotter.plot_cluster_statistics(cluster_stats=cluster_stats, path=path, logger=logger, title=title)
        #
        #     # plot gaussian plot
        #     path = f"{root}/gaussian_plot_{args.n_clusters}_{args.genres}.pdf"
        #     title = f"Gaussian mixture model cluster boundaries with {signal_processor} applied"
        #     plotter.plot_gmm(gmm=metric_l.gaussian_model, X=latent_space, path=path, logger=logger, labels=y_pred, title=title)
