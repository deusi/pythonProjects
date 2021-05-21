from sklearn.datasets import load_boston, load_digits
import numpy as np


def prepare_boston(pct):
    boston = load_boston()
    y = boston.target
    X = boston.data
    p = np.percentile(y, pct)
    y_new = np.where(y >= p, 1, 0)

    # k is hardcoded because we manually added two classes
    k = np.array([0,1])
    d = X.shape[1]
    return X, y_new, k, d


def prepare_boston50():
    return prepare_boston(50)


def prepare_boston25():
    return prepare_boston(25)


def prepare_digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    k = np.array(digits.target_names)
    d = X.shape[1]

    # add random noise to avoid singular matrix
    G = np.random.normal(0, 0.000001, X.size).reshape(X.shape)
    X = X + G
    return X, y, k, d

