from matplotlib import pyplot as plt
from sklearn.datasets import load_digits, make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.7, random_state=10)

cost_function = []
k_list = range(1,11)

for k in k_list:
    model = KMeans(n_clusters=k)
    model.fit(X)

    cost_function.append(abs(model.score(X)))

    pred = model.predict(X)


plt.plot(k_list,cost_function, "r")
plt.show()
