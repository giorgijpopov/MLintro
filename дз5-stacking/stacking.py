from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def stacking (estimator, X, y, Xt, k, method):
    kf = KFold (n_splits = k, shuffle = True, random_state = 0)
    if method == 'predict_proba':
        all_classes = np.unique (y)
        classes_indexes = {}
        for i in range (all_classes.size):
            classes_indexes[all_classes[i]] = i

        sX = np.zeros((X.shape[0], all_classes.size))
        sXt = np.zeros((Xt.shape[0], all_classes.size))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.fit (X_train, y_train)
            classes = estimator.classes_

            pred_proba = estimator.predict_proba (X_test)
            for j in range (pred_proba.shape[1]):
                sX[test_index, classes_indexes[classes[j]]] = pred_proba[:, j]

            pred_proba = estimator.predict_proba(Xt)
            for j in range(pred_proba.shape[1]):
                sXt[:, classes_indexes[classes[j]]] += pred_proba[:, j]

        sXt /= k
        return sX, sXt

    sX = np.zeros (X.shape[0])
    sXt = np.zeros (Xt.shape[0])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit (X_train, y_train)
        sX[test_index] = estimator.predict (X_test)
        sXt += estimator.predict (Xt)

    sXt /= k
    return sX, sXt

X = np.random.randint (2, size = (100, 4))
y = np.zeros (100)
for i in range (X.shape[0]):
    y[i] = X[i].sum ()

Xt = np.random.randint (2, size = (20, 4))

linreg = LinearRegression ()

sX, sXt = stacking (linreg, X, y, Xt, 4, 'predict')
print (sX, sXt)

for i in range (y.size):
    y[i] = y[i] % 2

clf = DecisionTreeClassifier ()
sX, sXt = stacking (clf, X, y, Xt, 4, 'predict_proba')
print (sX, sXt)


