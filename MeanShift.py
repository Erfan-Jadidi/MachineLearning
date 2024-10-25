import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import os

# Set the number of cores to use
os.environ['LOKY_MAX_CPU_COUNT'] = "4"

centers = [[1, 1], [5, 5], [9, 1], [-3,-3]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=2)

# تخمین پهنای باند (bandwidth)
bandwidth = estimate_bandwidth(X, quantile=0.2)

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))
print(f'Number of clusters: {n_clusters_}')

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='black', marker='x')
plt.title(f'Mean Shift Clustering with {n_clusters_} clusters')
plt.show()

