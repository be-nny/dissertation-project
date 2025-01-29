import matplotlib
import numpy as np
import pandas as pd
import squarify
from kneed import KneeLocator

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

import plot_lib

matplotlib.use('TkAgg')

def plot_cluster_statistics(cluster_stats: dict, path: str, logger) -> None:
    """
    Creates a figure with a set of pie chart subplots demonstrating which clusters have what genre in them.

    :param cluster_stats: cluster statistics
    :param path: path to save figure
    :param logger: logger
    """

    for cluster_key, values in cluster_stats.items():
        labels = []
        sizes = []
        for key, value in values.items():
            labels.append(key)
            sizes.append(cluster_stats[cluster_key][key])

        category_codes, unique_categories = pd.factorize(np.array(list(values.keys())))
        colours = [plot_lib.CMAP(code) for code in category_codes]

        total = sum(sizes)
        percentages = [round(s/total, 2) for s in sizes]
        labels = [f"{l}: {p}" for l, p in zip(labels, percentages)]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()

        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colours,
            ax=ax,
            text_kwargs={
                'color': 'black',
                'fontsize': 9,
                'fontweight': 'bold'
            },
            pad=True,
        )
        ax.set_title(f"Cluster Statistics for Cluster {cluster_key}")
        plt.show()


def plot_2d_kmeans_boundaries(latent_space: np.ndarray, kmeans, logger, path: str, genre_filter: str, h: float = 0.02) -> None:
    """
    Plots the kmeans decision boundaries

    :param latent_space: 2D latetn space
    :param kmeans: KMeans object
    :param logger: logger
    :param path: path to save
    :param genre_filter: genre filter
    :param h: step size for the grid used to create the mesh for plotting the decision boundaries
    """

    # colour map
    colors = [plot_lib.CMAP(i / (kmeans.n_clusters - 1)) for i in range(kmeans.n_clusters)]
    cmap = ListedColormap(colors)

    # plot the decision boundary
    x_min, x_max = latent_space[:, 0].min() - 1, latent_space[:, 0].max() + 1
    y_min, y_max = latent_space[:, 1].min() - 1, latent_space[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # obtain labels for each point in mesh. Use last trained model.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(grid_points)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    img = plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=cmap, aspect="auto", origin="lower")
    plt.plot(latent_space[:, 0], latent_space[:, 1], "k.", markersize=2)

    # plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
    plt.title(f"K-means Clustering Boundaries \n (genres: {genre_filter})")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # adding the colour bar
    colorbar = plt.colorbar(img, ticks=range(kmeans.n_clusters))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(kmeans.n_clusters)])

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_eigenvalues(path, pca_model, logger) -> None:
    plt.plot([i for i in range(1, pca_model.n_components + 1)], pca_model.explained_variance_, marker="o", linestyle="-", label="Eigenvalues")
    plt.xlabel("Number of Components")
    plt.ylabel("Eigenvalues (log)")
    plt.yscale("log")
    plt.title("PCA Eigenvalues (log scale)")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved plot '{path}'")

def plot_3D(latent_space: np.ndarray, y_true: np.ndarray, path: str, logger, genre_filter: str, loader) -> None:
    """
    Plots a 3 dimensional latent representation in 3D space

    :param latent_space: 3D latent space
    :param y_true: true labels
    :param path: path to save
    :param logger: logger
    :param genre_filter: genre filters
    :param loader: dataset loader
    :return: None
    """

    unique_labels = np.unique(y_true)
    str_labels = loader.decode_label(unique_labels)
    print(latent_space.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], c=y_true, cmap=plot_lib.CMAP, alpha=0.7, s=10)
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

def plot_2D(latent_space: np.ndarray, y_true: np.ndarray, path: str, logger, genre_filter: str, loader) -> None:
    """
    Plots a 2 dimensional latent representation in 2D space

    :param latent_space: 2D latent space
    :param y_true: true labels
    :param path: path to save
    :param logger: logger
    :param genre_filter: genre filters
    :param loader: dataset loader
    :return: None
    """

    unique_labels = np.unique(y_true)
    str_labels = loader.decode_label(unique_labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(latent_space[:, 0], latent_space[:, 1], c=y_true, cmap=plot_lib.CMAP, alpha=0.7, s=10)
    colour_bar = plt.colorbar(scatter, ax=ax, label="Cluster Labels")
    colour_bar.set_ticks(unique_labels)
    colour_bar.set_ticklabels(str_labels)

    ax.set_title(f"2D Visualisation of Latent Space (genres: {genre_filter})")
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.grid(False)

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))

def plot_gmm(gmm, X, labels, path, logger, ax=None):
    colors = [plot_lib.CMAP(i / (gmm.n_components - 1)) for i in range(gmm.n_components)]
    cmap = ListedColormap(colors)

    ax = ax or plt.gca()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap=cmap, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    colorbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(gmm.n_components))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticks(np.arange(gmm.n_components))
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(gmm.n_components)])

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")

def plot_inertia(latent_space, logger, path, max_clusters=20, n_genres=10):
    inertia = []
    k_values = range(1, max_clusters + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(latent_space)
        inertia.append(kmeans.inertia_)

    kneed_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
    elbow_k = kneed_locator.knee
    elbow_inertia = inertia[elbow_k - 1]

    plt.figure(figsize=(8, 6))
    plt.axvline(x=elbow_k, color="red", linestyle="--", label=f"Elbow: k={elbow_k}")
    plt.axhline(y=elbow_inertia, color="green", linestyle="--", label=f"Inertia: {elbow_inertia:.2e}")
    plt.plot(k_values, inertia, color="blue", marker="o", linestyle="-")
    plt.plot(n_genres, inertia[n_genres-1], color="red", marker="o", label=f"Total Genres: {n_genres}")

    plt.xticks(k_values)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Score")
    plt.title("KMeans Inertia")
    plt.legend()
    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")