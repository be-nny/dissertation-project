import os.path
from functools import partial

import matplotlib.pyplot as plt

from model.utils import CustomPoint
from plot_lib import *

def _draw_ellipse(position, covariance, ax=None, **kwargs) -> None:
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

def interactive_gmm(gmm, data_points: list[CustomPoint], title: str, path: str) -> (plt.Axes, plt.Figure):
    """
    Plot Gaussian Mixture Model with ellipses around points.
    """
    fig, ax = plt.subplots()

    x = [p.x for p in data_points]
    y = [p.y for p in data_points]
    labels = [p.y_pred for p in data_points]

    colours = [CMAP(i / (gmm.n_components - 1)) for i in range(gmm.n_components)]
    cmap = ListedColormap(colours)

    scatter = ax.scatter(x, y, c=labels, s=10, cmap=cmap, zorder=2, picker=5)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        _draw_ellipse(pos, covar, alpha=w * w_factor)

    colorbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(gmm.n_components))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticks(np.arange(gmm.n_components))
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(gmm.n_components)])

    plt.title(title)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")

    callback = partial(_show_nearest_neighbours, fig=fig, ax=ax, data_points=data_points, path=path)
    fig.canvas.mpl_connect('pick_event', callback)
    plt.savefig(path, bbox_inches='tight')
    return ax, fig

def interactive_kmeans(kmeans, data_points: list[CustomPoint], title: str, path: str, h: float = 0.02) -> (plt.Axes, plt.Figure):
    """
    Plots the kmeans decision boundaries

    :param title: title
    :param data_points: 2D latent space points
    :param kmeans: KMeans object
    :param path: path to save
    :param h: step size for the grid used to create the mesh for plotting the decision boundaries
    """
    fig, ax = plt.subplots()

    x = np.array([p.x for p in data_points])
    y = np.array([p.y for p in data_points])

    # Colour map
    colour_map = pypalettes.load_cmap("Benedictus")
    colours = [colour_map(i / (kmeans.n_clusters - 1)) for i in range(kmeans.n_clusters)]
    cmap = ListedColormap(colours)

    # Plot the decision boundary
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(grid_points).reshape(xx.shape)

    img = ax.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                     cmap=cmap, aspect="auto", origin="lower")

    # Make scatter points pickable
    ax.scatter(x, y, color='black', s=10, picker=True)

    # Plot the centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])

    # Colour bar
    colorbar = fig.colorbar(img, ax=ax, ticks=range(kmeans.n_clusters))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(kmeans.n_clusters)])

    ax.set_title(title)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")

    # Event connection
    fig.canvas.mpl_connect('pick_event', partial(_show_nearest_neighbours, fig=fig, ax=ax, data_points=data_points, path=path))

    fig.savefig(path, bbox_inches='tight')
    return ax, fig


def _show_nearest_neighbours(event, fig, ax, data_points, path):
    """
    Shows the nearest neighbours when a point is clicked on the canvas

    :param event: click event
    :param fig: figure
    :param ax: axes
    :param data_points: plotted data points
    """
    ind = event.ind
    for i in ind:
        for neighbour_data in data_points[i].nearest_neighbours:
            point = list(neighbour_data)[1]
            name = list(neighbour_data)[2]

            ax.annotate(f"{name}", (point[0], point[1]), textcoords="offset points", xytext=(0,3), ha='center', fontsize=8, alpha=0.5)
            ax.plot([data_points[i].x, point[0]], [data_points[i].y, point[1]], color='black')
            root = os.path.dirname(path)

        file_name = data_points[i].nearest_neighbours[0][2]

        picked_dir = os.path.join(root, "selected_nearest_neighbours")
        if not os.path.exists(picked_dir):
            os.makedirs(picked_dir)
        file_path = os.path.join(picked_dir, f"{file_name}_nearest_neighbours.pdf")
        plt.savefig(file_path, bbox_inches='tight')
    fig.canvas.draw()
