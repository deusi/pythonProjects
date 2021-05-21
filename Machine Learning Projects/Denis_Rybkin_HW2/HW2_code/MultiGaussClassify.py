import numpy as np
import numpy.linalg as la


def my_mean(X):
    # Returns the means of the columns of X
    n = np.float64(X.shape[0])
    return X.sum(axis=0) / n


def my_cov(X):
    # Returns the sample covariance matrix for X
    mu = np.ones(np.shape(X)) * my_mean(X)
    n = np.float64(np.shape(X)[0])
    return (1 / (n - 1)) * (X - mu).T.dot(X - mu)


class GaussianDiscriminant:

    def __init__(self, label, X, n, diag, def_cov, def_means, def_prior, d):
        # Keep the class label
        self.label = label

        # Initialize the default values, according to the write-up
        self.S = def_cov
        self.mu = def_means
        self.prior = def_prior
        self.d = d

        # Get the actual value for covariance matrix
        self.S = my_cov(X)

        # Turn covariance matrix into the diagonal one if requested
        if diag:
            self.S = np.diag(np.diag(self.S))

        # Calculate the inverse and determinant of S for later use
        self.S_inv = la.inv(self.S)
        self.S_det = la.det(self.S)

        # Properly initialize means and prior
        self.mu = my_mean(X)
        self.prior = X.shape[0] / np.float64(n)

    def discriminant(self, X):
        # Compute the determinant based on the formula from the hw2
        # (expanded version provided in lecture slides)
        # The constant can be removed, as it doesn't play a significant role, but I decided to keep it anyway
        return (
                (-1/2) * ((X - self.mu).T.dot(self.S_inv))
                .dot(X - self.mu) -
                (-1 / 2) * np.log(self.S_det)
                + np.log(self.prior)
                - (self.d/2) * np.log(2 * 3.1416)
                )


class MultiGaussClassify:

    def __init__(self, k, d, diag=False):
        # For keeping the discriminants
        self.classes = []

        self.k = k
        self.d = d
        self.diag = diag

        # We are required to initialize the default values
        self.covariance = np.identity(d)
        self.means = np.zeros(d)
        self.prior = np.float64(1/len(k))

    def fit(self, X, y):
        # Create new discriminants for each fit
        self.classes = []

        n = X.shape[0]

        # Create and initialize one discriminant per class
        for c in self.k:
            # Separate out the data for this class
            X_class = X[np.where(y == c)[0]]
            X_class.reshape(X_class.shape[0], self.d)

            discriminant = GaussianDiscriminant(c, X_class, n, self.diag, self.covariance, self.means, self.prior, self.d)

            self.classes.append(discriminant)

    def predict(self, X):

        y_predict = []

        for i in np.arange(X.shape[0]):
            scores = []

            # Evaluate the discriminant for each class
            for cls in self.classes:
                s = cls.discriminant(X[i].T)
                scores.append(s)
            # Determine the highest score
            i = np.argmax(scores)
            # Return the label for the class with the highest score
            y_predict.append(self.classes[i].label)

        return np.array(y_predict)
