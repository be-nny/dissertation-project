import os

from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

import config
import optuna

from sklearn.metrics import silhouette_score
from mat_logger import mat_logger
from mat_models import dim_rdct as dr, utils

config = config.Config(path="config.yml")
logger = mat_logger.get_logger()

# loading all data (combining test and train splits together)
dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="e9d165", logger=logger)
figures_path = os.path.join(dataset_loader.get_directory(), "figures")

data, labels = dataset_loader.get_data()

# scaling the data
scaler = StandardScaler(with_mean=False)
data = scaler.fit_transform(data)

# creating PCA model
pca_model = dr.PCAModel(
    logger=logger,
    data=data,
    genre_labels=labels,
    n_components=100,
    figures_path=figures_path
).create()

pca_embeddings = pca_model.get_embeddings()
pca_model.visualise()

# optimise umap parameters
def umap_objective(trial):
    n_neighbours = trial.suggest_int("n_neighbors", 2, 100)
    min_dist = trial.suggest_float("min_dist", 0.1, 1.99)
    n_components = trial.suggest_int("n_components", 3, 75)
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "manhattan", "hamming", "chebyshev"])
    learning_rate = trial.suggest_float("learning_rate", 0.1, 10.0)
    n_epochs = trial.suggest_int("n_epochs", 50, 1000)
    negative_sample_rate = trial.suggest_int("negative_sample_rate", 1, 10)
    repulsion_strength = trial.suggest_int("repulsion_strength", 1, 20)

    umap_model = dr.UmapModel(
        logger=logger,
        random_state=42,
        data=pca_embeddings,
        genre_labels=labels,
        figures_path=figures_path,
        n_neighbors=n_neighbours,
        min_dist=min_dist,
        spread=2,
        metric=metric,
        n_components=n_components,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
    ).create()

    embeddings = umap_model.get_embeddings()

    silhouette = silhouette_score(embeddings, labels)
    # trust = trustworthiness(data, embeddings, n_neighbors=10)
    # score = 0.7 * silhouette + 0.3 * trust

    return silhouette

study = optuna.create_study(direction="maximize")
study.optimize(umap_objective, n_trials=100)

# Get the best parameters
print("Best Parameters:", study.best_params)
print("Best Silhouette Score:", study.best_value)
