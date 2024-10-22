import time

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

result_list = []

for samples in [100, 500, 1000]:
    for features in [2, 4]:
        for stds in [1, 5, 7]:
            X, y = make_blobs(
                n_samples=samples,
                n_features=features,
                cluster_std=stds,
                random_state=32)

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=32)

            t1 = time.time()
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(x_train, y_train)
            train_time = time.time()- t1

            t2 = time.time()
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            test_time = time.time() - t2

            print(f"n={samples:6}, features={features:3}, std={stds:3}, accuracy={acc:10}, train_time={train_time}, test_time={test_time}")
            result_list.append({"n": samples, "features": features, "std": stds, "accuracy": acc, "train_time":train_time, "test_time":test_time})

df = pd.DataFrame(result_list)
df.to_excel("result.xlsx")
