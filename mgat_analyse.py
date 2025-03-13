import argparse
import os.path
from random import choices

import numpy as np

import model
import logger

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import *
from plot_lib.plotter import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from model import utils
from model import models

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# arguments parser
parser = argparse.ArgumentParser(prog='Music Genre Analysis Tool (MGAT) - Analyis', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-u", "--uuid", help="this takes a comma seperated list of uuids to load (e.g., 19ee37,58bd65)")
parser.add_argument("-n", "--n_clusters", type=int, help="number of clusters")
parser.add_argument("-g", "--genres", help="takes a comma seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")
parser.add_argument("-i", "--info", action="store_true", help="returns a list of available datasets to use")
parser.add_argument("-t", "--cluster_type", choices=["gmm", "kmeans"])

parser.add_argument("-cr", "--correlation", action="store_true", help="plots the n-nearest neighbours correlation")
parser.add_argument("-sh", "--shannon", action="store_true", help="plots the average shannon entropy values against as the number of clusters is increased")
parser.add_argument("-ei", "--eigen", type=int, help="plots the eigenvalues obtained after performing PCA. takes value for max n_components")
parser.add_argument("-vi", "--visualise", help="plots 2D latent space after applying a dimensionality reduction technique", choices=["umap", "pca", "tsne"])
parser.add_argument("-cl", "--classifier", help="plots the comparison of the accuracy between a set of multi-classifiers and different preprocessed datasets", choices=["umap", "pca", "tsne"])

BATCH_SIZE = 512

def _load_uuids(dataset_uuids, cluster_type):
    datasets = {}

    for uuid in dataset_uuids:
        with utils.ReceiptReader(filename=os.path.join(config.OUTPUT_PATH, f'{uuid}/receipt.json')) as receipt:
            signal_processor = receipt.signal_processor
            segment_duration = receipt.seg_dur

        # create a dataloader
        n_genres, genre_filter = get_genre_filter("all")
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=uuid, logger=logger, batch_size=model.BATCH_SIZE, verbose=False)
        loader.load(split_type="all", normalise=True, genre_filter=genre_filter, flatten=True)

        # create metric learner
        metric_leaner = models.MetricLeaner(loader=loader, n_clusters=n_genres, cluster_type=cluster_type)
        metric_leaner.create_latent()
        latent_space, y_pred, y_true = metric_leaner.get_latent(), metric_leaner.get_y_pred(), metric_leaner.get_y_true()

        inv_covar = None
        if cluster_type == "gmm":
            covar = metric_leaner.cluster_model.covariances_[0]
            inv_covar = np.linalg.inv(covar)

        datasets.update({uuid: {"signal_processor": signal_processor, "segment_duration": segment_duration, "latent_space": latent_space, "y_pred": y_pred, "y_true": y_true, "inv_covar": inv_covar, "loader": loader, "cluster_model": metric_leaner.cluster_model}})
    return datasets

def _make_save_dir(name):
    save_dir = os.path.join(root, name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return save_dir

if __name__ == "__main__":
    args = parser.parse_args()

    metrics = {}
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    root = os.path.join(config.OUTPUT_PATH, "_analysis")
    if not os.path.exists(root):
        os.mkdir(root)

    if args.info:
        show_info(logger, config)

    if args.eigen:
        if args.uuid is None:
            parser.error("missing flags. '-u,--uuid' must be set")

        save_dir = _make_save_dir("eigenvalue-plots")

        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
        pca_model = PCA(n_components=args.eigen)

        batch_loader = loader.load(split_type="all", normalise=True)
        data, y_true = loader.all()
        pca_model.fit(data)
        path = os.path.join(save_dir, f"{args.uuid}_{loader.signal_processor}_eigenvalues.pdf")

        title = f"PCA Eigenvalues (log scale) with {loader.signal_processor} applied \n Total features: {pca_model.n_features_in_}"
        plot_eigenvalues(path=path, pca_model=pca_model, title=title)
        logger.info(f"Saved plot '{path}'")

    if args.visualise:
        if args.uuid is None or args.genres is None:
            parser.error("missing flags. '-u,--uuid', '-g,--genres' must be set.")

        save_dir = _make_save_dir("latent-plots")

        _, genre_filter = get_genre_filter(args.genres)
        loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
        loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

        data, y_true = loader.all()
        dim_model = models.get_dim_model(args.visualise)
        latent = dim_model.fit_transform(data)

        path = os.path.join(save_dir, f"{args.uuid}_{loader.signal_processor}_{args.visualise.lower()}_{args.genres}_visualisation_2D.pdf")
        title = f"Latent Space in 2D with {loader.signal_processor} applied using {args.visualise.lower()}"
        plot_2D(latent_space=latent, y_true=y_true, path=path, genre_filter=args.genres, loader=loader, title=title)
        logger.info(f"Saved plot '{path}'")

    if args.correlation:
        if args.uuid is None or args.cluster_type is None:
            parser.error("missing flags. '-u,--uuid', '-t,--cluster_type' must be set.")

        save_dir = _make_save_dir("correlation")

        uuids = args.uuid.split(",")
        logger.info("Loading datasets...")
        datasets = _load_uuids(uuids, args.cluster_type)
        dataset_items = tqdm(datasets.items(), desc="Correlation accuracy...", unit="dataset")
        for uuid, data in dataset_items:
            latent_space, y_pred, y_true, inv_covar, sp, sd = data["latent_space"], data["y_pred"], data["y_true"], data["inv_covar"], data["signal_processor"], data["segment_duration"]
            plot_correlation_accuracy(latent_space=latent_space, y_true=y_true, covariance_mat=inv_covar, label=f"{sp}_{sd}")

        plt.title(f"Correlation Accuracy Comparison for {args.type}")
        correlation_accuracy_plot_path = os.path.join(save_dir, f"correlation_accuracy_{args.uuid}_{args.cluster_type}.pdf")
        plt.savefig(correlation_accuracy_plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot '{correlation_accuracy_plot_path}'")

    if args.shannon:
        if args.uuid is None or args.cluster_type is None:
            parser.error("missing flags. '-u,--uuid', '-t,--cluster_type' must be set.")

        save_dir = _make_save_dir("shannon-entropy")
        max_clusters = 20
        uuids = args.uuid.split(",")
        logger.info("Loading datasets...")
        datasets = _load_uuids(uuids, args.cluster_type)
        dataset_items = tqdm(datasets.items(), desc="Calculating Shannon Entropy Averages...", unit="dataset")
        for uuid, data in dataset_items:
            y_true, loader, cluster_model, latent_space, sp, sd = data["y_true"], data["loader"], data["cluster_model"], data["latent_space"], data["signal_processor"], data["segment_duration"]

            entropy_values = []
            for i in range(2, max_clusters + 1):
                if args.cluster_type == "gmm":
                    cluster_model.n_components = i
                else:
                    cluster_model.n_clusters = i

                y_pred = cluster_model.predict(latent_space)

                # work out the shannon entropy for each cluster
                c_shan_entropy = 0
                cluster_stats = utils.cluster_statistics(y_true, y_pred, loader)
                for cluster_key, values in cluster_stats.items():
                    labels = []
                    sizes = []
                    for key, value in values.items():
                        labels.append(key)
                        sizes.append(cluster_stats[cluster_key][key])

                    sizes = np.array(sizes)
                    probs = [s/sizes.sum() for s in sizes]
                    entropy = -np.sum(probs * np.log(probs))
                    c_shan_entropy += entropy

                # average these entropy values
                avg_c_shan_entropy = c_shan_entropy/i
                entropy_values.append(avg_c_shan_entropy)

            plot_shannon_entropy(n_clusters=[n for n in range(2, max_clusters + 1)], avg_shannon_vals=entropy_values, label=f"{sp}_{sd}")

        plt.title(f"Average Shannon Entropy Comparison for {args.cluster_type}")
        entropy_path = os.path.join(save_dir, f"entropy_averages_{args.uuid}_{args.cluster_type}.pdf")
        plt.savefig(entropy_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot '{entropy_path}'")

    if args.classifier:
        if args.uuid is None or args.genres is None:
            parser.error("missing flags. '-u,--uuid', '-g,--genres' must be set.")

        save_dir = _make_save_dir("classifier")

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(random_state=42),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

        datasets = args.uuid.split(",")
        bar_plot_scores = {}
        for dataset in datasets:
            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=dataset, logger=logger, batch_size=BATCH_SIZE)
            dim_model = models.get_dim_model(args.classifier)

            # training
            loader.load(split_type="train", normalise=True, genre_filter=genre_filter, flatten=True)
            train_data, y_train = loader.all()
            x_train = dim_model.fit_transform(train_data)

            # testing
            loader.load(split_type="test", normalise=True, genre_filter=genre_filter, flatten=True)
            test_data, y_test = loader.all()
            x_test = dim_model.fit_transform(test_data)

            scores = []
            for name, classifier in zip(names, classifiers):
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                scores.append(acc)

            bar_plot_scores.update({f"{loader.signal_processor}_{loader.segment_duration}": scores})

        classifier_path = os.path.join(save_dir, f"classifier_accuracy_{args.uuid}")
        plot_classifier_scores(bar_plot_scores, names, classifier_path)
