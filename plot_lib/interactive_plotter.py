from functools import partial

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

def interactive_gmm(gmm, data_points: list[CustomPoint], title) -> (plt.Axes, plt.Figure):
    """
    Plot Gaussian Mixture Model with ellipses around points.
    """

    x = [p.x for p in data_points]
    y = [p.y for p in data_points]
    labels = [p.y_pred for p in data_points]

    colours = [CMAP(i / (gmm.n_components - 1)) for i in range(gmm.n_components)]
    cmap = ListedColormap(colours)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=labels, s=5, cmap=cmap, zorder=2, picker=5)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        _draw_ellipse(pos, covar, alpha=w * w_factor)

    colorbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(gmm.n_components))
    colorbar.set_label('Cluster Labels', rotation=270, labelpad=15)
    colorbar.set_ticks(np.arange(gmm.n_components))
    colorbar.set_ticklabels([f"Cluster {i}" for i in range(gmm.n_components)])

    plt.title(title)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")

    callback = partial(on_click, fig=fig, ax=ax, data_points=data_points)
    fig.canvas.mpl_connect('pick_event', callback)

    return ax, fig

def on_click(event, fig, ax, data_points):
    ind = event.ind
    print("You clicked on point(s):", ind)
    for i in ind:
        for neighbour_data in data_points[i].nearest_neighbours:
            point = list(neighbour_data)[1]
            name = list(neighbour_data)[2]

            ax.annotate(f"{name}", (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            ax.plot([data_points[i].x, point[0]], [data_points[i].y, point[1]], color='black')

    fig.canvas.draw()
