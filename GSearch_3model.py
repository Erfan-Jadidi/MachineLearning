import time

import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml


def time_it(function_name):
    def inner(*args, **kwargs):
        start = time.time()
        result = function_name(*args, **kwargs)
        end = time.time()
        return end - start, result

    return inner


def create_data(samples, features, stds, train_size):
    X, y = make_blobs(
        n_samples=samples,
        n_features=features,
        cluster_std=stds,
        random_state=32)

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=32)
    return x_train, x_test, y_train, y_test


@time_it
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


@time_it
def evaluate_model(model, x_test, y_test, scoring="accuracy"):
    y_pred = model.predict(x_test)

    match scoring:
        case "accuracy":
            score = accuracy_score(y_test, y_pred)
        case "precision":
            score = precision_score(y_test, y_pred)
        case "recall":
            score = recall_score(y_test, y_pred)
        case "f1-score":
            score = f1_score(y_test, y_pred)
    return score


result_list = []

for model in [KNeighborsClassifier(n_neighbors=3), SVC(kernel="rbf"), LogisticRegression()]:
    for samples in [100, 500, 1000, 5000, 10000, 50000, 100000]:
        for features in [2, 4, 10]:
            for stds in [1, 1.5, 3, 5, 7, 12]:
                x_train, x_test, y_train, y_test = create_data(samples, features, stds, 0.8)

                train_time, model = train_model(model, x_train, y_train)
                test_time, score = evaluate_model(model, x_test, y_test, scoring="accuracy")

                if isinstance(model, KNeighborsClassifier):
                    param = {
                        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'n_neighbors' : [3, 4, 5, 6, 10]
                    }
                elif isinstance(model, SVC):
                    param = {
                        'C' : [0.01, 0.1, 1, 10, 100, 1000], 
                        'kernel' : ['linear', 'poly', 'rbf','sigmoid']
                    }
                else:
                    param = {
                    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                } 

                GSearch = GridSearchCV(estimator=model, param_grid=param, cv=5, verbose=2)

                GSearch.fit(x_train, y_train)

                
                # print(
                #     f"model={model.__class__.__name__:20}, n={samples:6}, features={features:3}, std={stds:3}, accuracy={score:10}, train_time={train_time}, test_time={test_time}")
                result_list.append(
                    {"model": model.__class__.__name__,
                     "n": samples,
                     "features": features,
                     "std": stds,
                     "accuracy": score,
                     "train_time": train_time,
                     "test_time": test_time,
                     "Best Estimator" : GSearch.best_estimator_,
                     "Best Parameter" :  GSearch.best_params_})

df = pd.DataFrame(result_list)
df.to_excel("F:\\ML\\ML 5\\ML 05\\result.xlsx")
