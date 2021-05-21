import numpy as np


def rand_proj(X, d):
    G = np.random.normal(0, 1, (64, d))
    X_tilde = X.dot(G)
    return X_tilde
