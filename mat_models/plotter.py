import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

matplotlib.use('TkAgg')

def plot_silhouette(path, data, labels, n_clusters):
    silhouette_vals = silhouette_samples(data, labels)
    silhouette_avg = silhouette_score(data, labels)

    plt.figure(figsize=(10, 6))
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        y_upper = y_lower + len(cluster_silhouette_vals)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette_vals,
            alpha=0.7, label=f"Cluster {i + 1}"
        )
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette Score")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Samples")
    plt.title("Silhouette Analysis")
    plt.legend()
    plt.savefig(path)