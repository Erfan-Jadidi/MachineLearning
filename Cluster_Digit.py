import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X,y = load_digits(return_X_y=True)
print(len(X))

pca = PCA(n_components=64)

X_pca = pca.fit_transform(X)
print(type(X_pca))
model  = KMeans(n_clusters=10)

model.fit(X_pca)
pred = model.predict(X_pca)


X_cluster_0 = X[pred == 0]
X_cluster_1 = X[pred == 1]
X_cluster_2 = X[pred == 2]

# print(type(X_cluster_0))
# print(X_cluster_0)


y_cluster_0 = y[pred == 0]
y_cluster_1 = y[pred == 1]
y_cluster_2 = y[pred == 2]


# # print(y_cluster_0)
# print(np.unique(y_cluster_0))
# # print(y_cluster_1)
# print(np.unique(y_cluster_1))
# # print(y_cluster_2)
# print(np.unique(y_cluster_2))

for i in range(10):
    print(np.unique(y[pred == i]))

# print(X_cluster_0.shape)
# print(X_cluster_1.shape)
# print(X_cluster_2.shape)
# print(y_cluster_0.shape)
# print(y_cluster_1.shape)
# print(y_cluster_2.shape)

# print(np.unique(y_cluster_1))