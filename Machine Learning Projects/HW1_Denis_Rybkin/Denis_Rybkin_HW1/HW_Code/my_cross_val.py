import numpy as np
from random import seed


def cross_validation_split(dataset, folds=10):
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    # randomize the set of data
    np.random.RandomState(seed()).shuffle(indices)

    fold_sizes = np.full(folds, n_samples // folds, dtype=np.int)
    fold_sizes[:n_samples % folds] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_arr = np.array([item for item in indices if item not in indices[start:stop]])
        yield test_arr, indices[start:stop]
        current = stop


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def my_cross_val(model, X, y, k=10):
    scores = []
    for train_index, test_index in cross_validation_split(X, k):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        scores.append(1 - get_score(model, X_train, X_test, y_train, y_test))
    return np.array(scores)
