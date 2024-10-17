from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("F:\ML\Project\diabetes.csv")

X = df.drop(columns="Outcome")

y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

solvers = {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}

for solver in solvers:


    model = LogisticRegression(max_iter=5000, solver=solver)

    model.fit(X_train, y_train)

    total_pred = accuracy_score(model.predict(X), y)

    train_pred = accuracy_score(model.predict(X_train), y_train)

    test_pred = accuracy_score(model.predict(X_test), y_test)



    print(f"{solver} : ", total_pred, train_pred, test_pred)