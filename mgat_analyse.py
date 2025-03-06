import argparse
import logger
import umap.umap_ as umap

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import *
from plot_lib.plotter import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from model import utils

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - EXPERIMENTS', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-u", "--uuid", help="UUID of the preprocessed dataset to use")
parser.add_argument("-n", "--n_clusters", type=int, help="number of clusters")
parser.add_argument("-g", "--genres", help="Takes a comma-seperated string of genres to use (e.g., jazz,rock,blues,disco) - if set to 'all', all genres are used")

parser.add_argument("-i", "--info", action="store_true", help="Returns a list of available datasets to use")
parser.add_argument("-e", "--eigen", type=int, help="Plots the eigenvalues obtained after performing PCA. Takes value for max n_components")
parser.add_argument("-k", "--kmeans", help="Plots 2D Kmeans Boundaries of UMAP or PCA. '-nc' must be set for the number of clusters. '-g' must also be set.")
parser.add_argument("-t", "--inertia", help="Plots number of clusters against kmeans inertia score for UMAP or PCA. '-g' must also be set. '-nc' must be set for the max. number of clusters")
parser.add_argument("-v", "--visualise", help="Plots 2D Latent Space of UMAP or PCA. '-g' must also be set.")
parser.add_argument("-cl", "--classifier", help="Uses a Random Forest model to classify the data")

BATCH_SIZE = 512

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

if __name__ == "__main__":
    args = parser.parse_args()

    metrics = {}
    config = config.Config(path=args.config)
    logger = logger.get_logger()

    experiments_dir_root = os.path.join(config.OUTPUT_PATH, "analysis")
    if not os.path.exists(experiments_dir_root):
        os.mkdir(experiments_dir_root)

    if args.info:
        show_info(logger, config)

    if args.uuid:
        if args.eigen:
            experiments_dir = os.path.join(experiments_dir_root, "eigenvalue-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            pca_model = PCA(n_components=args.eigen)

            batch_loader = loader.load(split_type="all", normalise=True)
            data, y_true = loader.all()
            pca_model.fit(data)
            path = os.path.join(experiments_dir, f"{args.uuid}_{loader.signal_processor}_eigenvalues.pdf")

            title = f"PCA Eigenvalues (log scale) with {loader.signal_processor} applied \n Total features: {pca_model.n_features_in_}"
            plot_eigenvalues(path=path, pca_model=pca_model, title=title)
            logger.info(f"Saved plot '{path}'")

        if args.kmeans:
            experiments_dir = os.path.join(experiments_dir_root, "kmeans-boundary-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = loader.all()
            dim_model = get_dim_model(args.kmeans)

            latent = dim_model.fit_transform(data).astype(np.float64)
            kmeans = KMeans(n_clusters=args.n_clusters)
            y_pred = kmeans.fit_predict(latent)

            # plot boundaries
            path1 = os.path.join(experiments_dir, f"{args.uuid}_{loader.signal_processor}_{args.kmeans.lower()}_{args.genres}_kmeans_boundaries_{args.n_clusters}.pdf")
            title = f"K-Means cluster boundaries with {loader.signal_processor} applied"
            plot_2d_kmeans_boundaries(kmeans=kmeans, latent_space=latent, path=path1, genre_filter=args.genres, title=title)
            logger.info(f"Saved plot '{path1}'")

            # plot genre spread
            stats = utils.cluster_statistics(y_true=np.array(y_true), y_pred=np.array(y_pred), loader=loader)
            path2 = os.path.join(experiments_dir, f"{args.uuid}_{loader.signal_processor}_{args.kmeans.lower()}_{args.genres}_tree_plot_{args.n_clusters}.pdf")
            plot_tree_map(cluster_stats=stats, path=path2)
            logger.info(f"Saved plot '{path2}'")

        if args.inertia:
            experiments_dir = os.path.join(experiments_dir_root, "kmeans-inertia-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            n_genres, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            loader.load(split_type="all", normalise=True, genre_filter=genre_filter)
            data, y_true = loader.all()
            dim_model = get_dim_model(args.inertia)

            latent_space = dim_model.fit_transform(data)
            max_clusters = args.n_clusters
            path = os.path.join(experiments_dir, f"{args.uuid}_{loader.signal_processor}_{args.inertia.lower()}_{args.genres}_kmeans_inertia.pdf")
            title = f"KMeans Inertia with {loader.signal_processor} applied"
            plot_inertia(latent_space=latent_space, max_clusters=max_clusters, n_genres=n_genres, path=path, title=title)
            logger.info(f"Saved plot '{path}'")

        if args.visualise:
            experiments_dir = os.path.join(experiments_dir_root, "latent-plots")
            if not os.path.exists(experiments_dir):
                os.mkdir(experiments_dir)

            _, genre_filter = get_genre_filter(args.genres)
            loader = utils.Loader(out=config.OUTPUT_PATH, uuid=args.uuid, logger=logger, batch_size=BATCH_SIZE)
            loader.load(split_type="all", normalise=True, genre_filter=genre_filter)

            data, y_true = loader.all()
            dim_model = get_dim_model(args.visualise)
            latent = dim_model.fit_transform(data)

            path = os.path.join(experiments_dir, f"{args.uuid}_{loader.signal_processor}_{args.visualise.lower()}_{args.genres}_visualisation_2D.pdf")
            title = f"Latent Space in 2D with {loader.signal_processor} applied using {args.visualise.lower()}"
            plot_2D(latent_space=latent, y_true=y_true, path=path, genre_filter=args.genres, loader=loader, title=title)
            logger.info(f"Saved plot '{path}'")

        if args.classifier:
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
                dim_model = get_dim_model(args.classifier)
                rf_model = RandomForestClassifier(random_state=0)

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

                bar_plot_scores.update({loader.signal_processor: scores})
            plot_classifier_scores(bar_plot_scores, names, "")