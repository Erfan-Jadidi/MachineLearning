from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import numpy as np
from sklearn.cluster import DBSCAN

# centers = np.array([(1,1), (10,10), (-5,-5)])

# X, y = make_blobs(
#     n_samples=400,
#     n_features=2,
#     centers=centers,
#     cluster_std=1.5,
# )
X,y = make_moons(n_samples=500, noise=0.05, random_state=10)

# plt.subplot(1,2,1)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.scatter(centers[:, 0], centers[:, 1], c="red")



# plt.subplot(1,2,2)
model = DBSCAN(eps=0.3, min_samples=8)
pred = model.fit_predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=pred)
#
# plt.show()


labels = model.labels_
noise_point = np.sum(labels == 1)
