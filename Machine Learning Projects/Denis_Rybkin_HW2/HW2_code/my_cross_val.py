import numpy as np

# Changed my_cross_val to use my own definition of .fit and .predict instead
# Also, polished it compared to my previous implementation


def split_partition(data, k):
    # Break the data into k parts
    return np.array_split(data, k)


def train_validation_split(folds, i):
    # Copy folds so we don't affect the caller
    fold = folds[:]
    validation = fold.pop(i)
    return np.concatenate(fold), validation


def my_accuracy_score(y1, y2):
    # Return success rate between two label arrays
    if len(y1.shape) == 1:
        y1 = y1.reshape(len(y1), 1)

    if len(y2.shape) == 1:
        y2 = y2.reshape(len(y2), 1)

    return np.sum(y1 == y2) / np.float64(y1.shape[0])


def train_model(method, train, validation):
    # Train a model on the given algorithm
    train_y = train[:, -1:]
    train_X = train[:, :-1]
    validation_y = validation[:, -1:]
    validation_X = validation[:, :-1]
    method.fit(train_X, train_y.ravel())
    y_pred = method.predict(validation_X)
    return my_accuracy_score(validation_y, y_pred)


def my_cross_val(method, X, y, k, shuffle=True):
    # Perform k-fold cross-validation
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    scores = []
    # Combine the features and labels
    data = np.append(X, y, axis=1)

    # Shuffle if requested
    if shuffle:
        np.random.shuffle(data)

    folds = split_partition(data, k)

    for i in range(len(folds)):
        train, validation = train_validation_split(folds, i)
        acc = train_model(method, train, validation)
        scores.append(acc)
    return scores
