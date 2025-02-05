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

def plot_gmm(gmm, X, labels, title, ax=None,new_data=None, new_label=None) -> None:
    """
    Plot Gaussian Mixture Model with ellipses around points.

    :param new_label: the labels for the new song segments in 'new_data'
    :param new_data: plots a song highlighted in a different colour
    :param title: title
    :param gmm: gaussian mixture model
    :param X: data
    :param labels: true labels
    :param ax: ax to draw figure on
    """
    colours = [CMAP(i / (gmm.n_components - 1)) for i in range(gmm.n_components)]
    cmap = ListedColormap(colours)

    ax = ax or plt.gca()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap=cmap, zorder=2)
    ax.axis('equal')

    if new_data is not None:
        ax.plot(new_data[:, 0], new_data[:, 1], c="purple", marker="o", linewidth=1, zorder=2, markersize=3, label=new_label)
        ax.scatter(new_data[0][0], new_data[0][1], marker="^", linewidth=1, zorder=3, s=3, label="start")
        ax.scatter(new_data[-1][0], new_data[-1][1], marker="s", linewidth=1, zorder=3, s=3, label="end")
        ax.legend()

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
