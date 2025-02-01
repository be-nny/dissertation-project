import matplotlib
import numpy as np
import pandas as pd
import squarify
from holoviews.plotting.bokeh.styles import marker

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

import plot_lib

matplotlib.use('TkAgg')

def draw_ellipse(position, covariance, ax=None, **kwargs) -> None:
    """
    Draw an ellipse with a given position and covariance

    :param position: position of the ellipse
    :param covariance: covariance of the ellipse
    :param ax: ax to draw the ellipse on
    :param kwargs: keyword arguments
    """

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

def plot_gmm(gmm, X, labels, path, logger, ax=None, new_data=None, new_label=None) -> None:
    """
    Plot Gaussian Mixture Model with ellipses around points.

    :param gmm: gaussian mixture model
    :param X: data
    :param labels: true labels
    :param path: path to save
    :param logger: logger
    :param ax: ax to draw figure on
    """

    colors = [plot_lib.CMAP(i / (gmm.n_components - 1)) for i in range(gmm.n_components)]
    cmap = ListedColormap(colors)

    ax = ax or plt.gca()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap=cmap, zorder=2)
    ax.axis('equal')

    if new_data is not None:
        ax.plot(new_data[:, 0], new_data[:, 1], c="purple", marker="o", linewidth=1, zorder=2, markersize=3, label=new_label)
        ax.scatter(new_data[0][0], new_data[0][1], marker="D", zorder=3, s=10, label="start")
        ax.scatter(new_data[-1][0], new_data[-1][1], marker="s", zorder=3, s=10, label="end")
        ax.legend()

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    colorbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(gmm.n_components))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticks(np.arange(gmm.n_components))
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(gmm.n_components)])

    plt.savefig(path)
    logger.info(f"Saved plot '{path}'")
    plt.close()

def plot_cluster_statistics(cluster_stats: dict, path: str, logger) -> None:
    """
    Creates a figure with a set of pie chart subplots demonstrating which clusters have what genre in them.

    :param cluster_stats: cluster statistics
    :param path: path to save figure
    :param logger: logger
    """
    n_cols = 4
    n_rows = int(np.ceil(len(cluster_stats) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, (cluster_key, values) in enumerate(cluster_stats.items()):
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

        ax = axes[i]
        ax.set_axis_off()

        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colours,
            ax=ax,
            text_kwargs={
                'color': 'black',
                'fontsize': 9,
                'fontweight': 'bold',
                'bbox': dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            },
            pad=True,
        )
        ax.set_title(f"Cluster Statistics for Cluster {cluster_key}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    logger.info(f"Saved plot '{path}'")
    plt.close()