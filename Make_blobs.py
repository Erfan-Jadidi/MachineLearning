from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt


X, y = make_blobs(n_samples=100, n_features=2, centers=5, cluster_std=5)

axis = plt.figure()

fig = axis.add_subplot(111, projection = '3d')

fig.scatter(X[:, 0], X[:, 1], y, c = y)

plt.show()