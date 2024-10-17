from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("Project/diabetes.csv")

X = df.drop(columns = "Outcome")
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression(max_iter=5000, warm_start=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

import pickle

with open("model.dat", "wb") as file:
    pickle.dump(model, file)


# ----------------------------------------------------------------

import pickle

with open("model.dat", "rb") as file:
    model = pickle.load(file)

print(model.predict([[9,145,88,34,165,30.3,0.771,53]]))