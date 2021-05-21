import numpy as np


def quad_proj(X):
    X_tilde = []
    for items in X:
        row = []
        for item in items:
            row.append(item)
        for item in items:
            row.append(item*item)
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items):
                row.append(items[i] * items[j])
                j += 1
            i += 1
        X_tilde.append(row)
    return np.array(X_tilde)
