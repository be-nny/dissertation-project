import os
import matplotlib
import networkx as nx
import umap
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

import mat_config

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from mat_logger import mat_logger
from mat_models import dim_rdct as dr, utils, plotter

matplotlib.use('TkAgg')

config = mat_config.Config(path="config.yml")
logger = mat_logger.get_logger()

dataset_loader = utils.Loader(out=config.OUTPUT_PATH, uuid="d1cf0c")
data, labels = dataset_loader.get_data()
figures_path = os.path.join(dataset_loader.get_directory(), "figures")

pca_model = dr.PCAModel(
    logger=logger,
    data=data,
    genre_labels=labels,
    n_components=35,
    figures_path=figures_path
).create()

pca_embeddings = pca_model.get_embeddings()
pca_model.visualise()

umap_model = dr.UmapModel(
    logger=logger,
    data=pca_embeddings,
    genre_labels=labels,
    figures_path=figures_path,
    n_neighbors=15,
    spread=20,
    min_dist=0.1,
    n_components=15,
    metric="cosine",
    learning_rate=0.8,
    local_connectivity=10
).create()
umap_model.visualise()

# gmm = GaussianMixture(n_components=len(dataset_loader.get_genres()), covariance_type="full", random_state=42)
# gmm.fit(umap_embeddings)
# gmm_labels = gmm.predict(umap_embeddings)
# plotter.plot_silhouette(os.path.join(figures_path, "gmm_silhouette.png"), data=data, labels=gmm_labels, n_clusters=gmm.n_components)
#
# kmeans = KMeans(n_clusters=len(dataset_loader.get_genres()))
# kmeans.fit(umap_embeddings)
# plotter.plot_silhouette(path=os.path.join(figures_path, "kmeans_silhouette.pdf"), data=data, labels=kmeans.labels_, n_clusters=kmeans.n_clusters)
#
# agglomerative_model = AgglomerativeClustering(n_clusters=len(dataset_loader.get_genres()))
# agglomerative_model.fit(umap_embeddings)
# plotter.plot_silhouette(path=os.path.join(figures_path, "agglomerative_silhouette.pdf"), data=data, labels=agglomerative_model.labels_, n_clusters=agglomerative_model.n_clusters)
#
# nmi_kmeans = normalized_mutual_info_score(labels_true=labels, labels_pred=kmeans.labels_)
# nmi_agg = normalized_mutual_info_score(labels_true=labels, labels_pred=agglomerative_model.labels_)
# nmi_gmm = normalized_mutual_info_score(labels_true=labels, labels_pred=gmm_labels)
#
# print(f"Kmeans NMI Score: {nmi_kmeans}")
# print(f"Agglomerative NMI Score: {nmi_agg}")
# print(f"Gaussian Mixture Model NMI Score: {nmi_gmm}")


