from sklearn.datasets import load_boston
import numpy as np


# prepare dataset for later usage
def prepare_boston(pct):
    boston = load_boston()
    y = boston.target
    x = boston.data
    p = np.percentile(y, pct)
    # changed the labels for SVM
    y1 = np.where(y >= p, 1, -1)
    return x, y1


def prepare_boston25():
    return prepare_boston(25)


def prepare_boston50():
    return prepare_boston(50)

