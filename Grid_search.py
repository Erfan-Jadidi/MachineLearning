from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=5, random_state=32)

params_svc = {
    'C' : [0.01, 0.1, 1, 10, 100, 1000], 
    'kernel' : ['linear', 'poly', 'rbf','sigmoid'],
}

params_lr = {
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}   
for est in [SVC(), LogisticRegression(max_iter=3000)]:
    if isinstance(est, SVC):
        param = params_svc
    else:
        param = params_lr
    GSearch = GridSearchCV(estimator=est, param_grid=param, cv=5, verbose=2)
    GSearch.fit(X, y)
    print(f'Best Parameters for {est.__class__.__name__}: {GSearch.best_params_}')