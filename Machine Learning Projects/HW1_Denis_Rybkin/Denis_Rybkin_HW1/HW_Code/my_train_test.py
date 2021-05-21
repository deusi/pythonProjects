import numpy as np
from random import seed


def my_train_test_split(dataset, pi, folds):
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    train_split = int(n_samples * pi)

    count = 0
    while count < folds:
        # randomize the set of data
        np.random.RandomState(seed()).shuffle(indices)
        yield indices[:train_split], indices[train_split:]
        count += 1


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def my_train_test(model, X, y, pi=0.75, k=10):
    scores = []
    for train_index, test_index in my_train_test_split(X, pi, k):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        scores.append(1 - get_score(model, X_train, X_test, y_train, y_test))
    return np.array(scores)