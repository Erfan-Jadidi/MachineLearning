from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.6)

X1, y1 = make_moons(n_samples=300, noise=0.1)

X2, y2 = make_circles(n_samples=300)

model = DBSCAN(eps = 0.1, min_samples=5)
pred = model.fit_predict(X2)

plt.subplot(1, 2, 1)
plt.scatter(X2[:, 0], X2[:, 1], c=y2)

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=pred)

plt.show()
