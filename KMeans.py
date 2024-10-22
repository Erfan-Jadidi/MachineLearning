from matplotlib import pyplot as plt
from sklearn.datasets import load_digits, make_blobs
from sklearn.cluster import KMeans

cost_function = []

X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.7, random_state=10)

model = KMeans(n_clusters=8)
model.fit(X)

cost_function.append(model.score(X))

pred = model.predict(X)

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=pred)
plt.show()