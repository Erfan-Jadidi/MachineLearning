from keras.datasets import mnist

from sklearn.svm import SVC

import numpy as np

from sklearn.metrics import classification_report

import cv2 as cv


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:600]
y_test = y_test[:600]

print(X_train.shape)

X_train = np.array([((x.astype(np.float32)).reshape(-1)) / 255.0 for x in X_train])
X_test = np.array([((x.astype(np.float32)).reshape(-1)) / 255.0 for x in X_test])

clf = SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

img = cv.imread("F:\\ML\\Project\\zero.png")

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img1 = np.array([img.astype(np.float32).reshape(-1) / 255.0])

print(clf.predict(img1))
cv.imshow('hi', img)
cv.waitKey(0)