import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

# Impute , DBSCAN, CLUSTERING

X, y = fetch_openml("mnist_784", return_X_y=True)

# StandardScaler(X)

# X,y = load_digits(return_X_y=True)

# Grid Search
model_1 = SVC(probability=True)
model_1.fit(X, y)
pred_proba = model_1.predict_proba(X)
pred_1 = model_1.predict(X)
print(classification_report(y, pred_1))

X_false = X[y != pred_1]
y_false = y[y != pred_1]

# manipulate
# Grid Search
model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(pred_proba, y)

pred = model_2.predict(pred_proba)
print(classification_report(y, pred))

# index Correct, Wrong
# SVM  1,2,3,8 +           4,5,6,7 -
# KNN  1,2,3,6,7,8 +       4,5,2 -

