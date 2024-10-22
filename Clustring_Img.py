import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv.imread("F:\\ML\\ML 6\\ML 06\\face.png", cv.IMREAD_GRAYSCALE)

vector_image = img.reshape(-1, 1)

model = KMeans(n_clusters=8)
model.fit(vector_image)
pred = model.predict(vector_image)

plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")

plt.subplot(2,2,3)
plt.hist(img.ravel(), bins=256)

plt.subplot(2,2,2)
plt.imshow(pred.reshape(img.shape), cmap="gray")

plt.subplot(2,2,4)
plt.hist(pred, bins=256)


plt.show()