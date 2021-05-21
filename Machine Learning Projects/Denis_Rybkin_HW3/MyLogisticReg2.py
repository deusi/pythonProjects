import numpy as np


def gradient_descent(w, iterations, eta, loss_grad):
    # decided to use batch gradient descent
    for i in range(0, iterations):
        loss, grad = loss_grad(w)
        w -= (eta * grad)
    return w


def scale_data(x, m=None, s=None):
    if m is None:
        m = x.mean(axis=0)
    if s is None:
        s = x.std(axis=0)
    x = (x - m) / s
    return x, m, s


def sigmoid(z):
    # using sigmoid given in the lecture, since the intersect term doesn't play a significant role
    return np.exp(z) / (1 + np.exp(z))


def logistic_loss_grad(w, x, y, reg):
    z = x.dot(w)
    p = sigmoid(z)
    loss = -np.sum(y * z - np.log(1 + np.exp(z))) + (reg / 2) * np.sum(w * w)
    grad = x.T.dot(p - y) + reg * w
    return loss, grad


def create_loss_grad(x, y, reg):
    def loss_grad(w):
        return logistic_loss_grad(w, x, y, reg)
    return loss_grad


class MyLogisticReg2:

    def __init__(self, d):
        # initialize parameters (w, w_0)
        self.w = np.zeros(d)

    def _train(self, x, y, reg, iterations, eta):
        # Scale the data
        x, self.m, self.s = scale_data(x)
        self.w = gradient_descent(self.w, iterations, eta, create_loss_grad(x, y, reg))

    def _prep_predict(self, x):
        x = np.copy(x)
        x = scale_data(x, self.m, self.s)[0]
        return x

    def fit(self, x, y):
        # these values seem to produce the best result
        # can be further tweaked for increased accuracy
        reg = 1.0
        iterations = 500
        eta = 0.01
        self._train(x, y, reg, iterations, eta)

    def predict(self, x):
        x = self._prep_predict(x)
        y_pred = []
        for i in np.arange(x.shape[0]):
            y_hat = sigmoid(self.w.T.dot(x[i]))
            if y_hat > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return np.array(y_pred)
