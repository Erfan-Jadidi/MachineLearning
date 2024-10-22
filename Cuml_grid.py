import time

import pandas as pd

from tensorflow.keras.datasets import mnist
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from cuml.svm import SVC
from cuml.metrics import accuracy_score
from cuml.neighbors import KNeighborsClassifier
from cuml.model_selection import train_test_split



(x_train, y_train), (x_test, y_test) = mnist.load_data()
result_list = []



for est in [KNeighborsClassifier(), LogisticRegression(), SVC()]:
  for data_count in [500,1000,5000,10000,30000,60000]:
    if isinstance(est, LogisticRegression):
      x_train = np.array([np.ravel(x.astype(np.float32)) / 255.0 for x in x_train])
      x_test = np.array([np.ravel(x.astype(np.float32)) / 255.0 for x in x_test])

      x_train = x_train[:data_count]
      y_train = y_train[:data_count]

      x_test = x_test[:data_count]
      y_test = y_test[:data_count]
    else:
      x_train = [np.ravel(x) for x in x_train]
      x_test = [np.ravel(x) for x in x_test]

      x_train = x_train[:data_count]
      y_train = y_train[:data_count]

      x_test = x_test[:data_count]
      y_test = y_test[:data_count]
    
    param_Lg = {"C": [0.01, 0.1, 0.5, 1, 5, 10 , 50, 100]}
    param_SVC = {"C": [0.01, 0.1, 0.5, 1, 5, 10 , 50, 100], "kernel" : ['linear', 'poly', 'rbf', 'sigmoid']}
    param_KNN = {"n_neighbors" : [3, 5, 7, 10, 12]}

    if isinstance(est, KNeighborsClassifier):
      param = param_KNN
    elif isinstance(est, SVC):
      param = param_SVC
    else:
      param = param_Lg
    
    
    gr_s_model = GridSearchCV(est, param, cv=5, verbose=2, scoring="accuracy")
    gr_s_model.fit(x_train, y_train)

    y_pred = gr_s_model.predict(x_test)
    score = accuracy_score(y_test, y_pred)

    result_list.append(
                    {"model": est.__class__.__name__,
                     "n": data_count,
                     "accuracy": score,
                     "Best Estimator" : gr_s_model.best_estimator_,
                     "Best Parameter" :  gr_s_model.best_params_})


best_result = max(result_list, key=lambda x: x['accuracy'])

print(f"Best Model: {best_result['model']}")
print(f"Accuracy: {best_result['accuracy']}")
print(f"Best Estimator: {best_result['Best Estimator']}")
print(f"Best Params: {best_result['Best Parameter']}")













# svm_model = SVC()
# Log_model = LogisticRegression()
# KNN_model = KNeighborsClassifier()
# for model in [KNeighborsClassifier(), SVC(), LogisticRegression()]:
#     for samples in [10, 100, 1000, 10000]:
#         for features in [2, 3, 4, 7, 10]:
#             for stds in [2, 3, 7, 9]:
#                 x_train, x_test, y_train, y_test = create_data(samples, features, stds, 0.8)

#                 train_time, model = train_model(model, x_train, y_train)
#                 test_time, score = evaluate_model(model, x_test, y_test, scoring="accuracy")

#                 if isinstance(model, KNeighborsClassifier):
#                     param = {
#                         'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                         'n_neighbors' : [3, 4, 5, 6, 10]
#                     }
#                 elif isinstance(model, SVC):
#                     param = {
#                         'C' : [0.01, 0.1, 1, 10, 100, 1000],
#                         'kernel' : ['linear', 'poly', 'rbf','sigmoid']
#                     }
#                 else:
#                     param = {
#                     'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
#                 }

#                 GSearch = GridSearchCV(estimator=model, param_grid=param, cv=5, verbose=2)

#                 GSearch.fit(x_train, y_train)

#                 # print()
#                 # print()
#                 # print("Best Score : ", GSearch)




#                 # print(
#                 #     f"model={model.__class__.__name__:20}, n={samples:6}, features={features:3}, std={stds:3}, accuracy={score:10}, train_time={train_time}, test_time={test_time}")
#                 result_list.append(
#                     {"model": model.__class__.__name__,
#                      "n": samples,
#                      "features": features,
#                      "std": stds,
#                      "accuracy": score,
#                      "train_time": train_time,
#                      "test_time": test_time,
#                      "Best Estimator" : GSearch.best_estimator_,
#                      "Best Parameter" :  GSearch.best_params_})

# df = pd.DataFrame(result_list)
# df.to_excel("F:\\ML\\ML 5\\ML 05\\result.xlsx")