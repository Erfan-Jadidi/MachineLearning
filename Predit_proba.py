from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("F:\ML\Project\diabetes.csv")

X = df.drop(columns = "Outcome")
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

model = LogisticRegression(max_iter = 5000, warm_start = True)

model.fit(X_train, y_train)

y_test_pred_proba = model.predict_proba(X_test)

print(y_test_pred_proba[:1])
print(y_test_pred_proba[:1, 1])