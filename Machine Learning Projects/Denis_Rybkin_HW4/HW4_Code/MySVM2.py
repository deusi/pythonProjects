import numpy as np


def compute_cost(W, X, Y, reg):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


def calculate_cost_gradient(W, X_batch, Y_batch, reg):
    # if only one example is passed
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)  # average
    return dw


# function to create a list containing mini-batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = 0
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class MySVM2:

    # Init functions
    def __init__(self, d, maxIterations, batch):
        self.w = 0.02*np.random.random_sample((d+1,)) - 0.01
        self.maxIterations = maxIterations
        self.epsilon = 0.0001
        self.eta = 0.01
        self.mean = 0
        self.std = 0
        # set the batch size
        self.batch = batch
        # manually set lambda to 5, as per requirements
        self.reg = 5

    # Fit the dataset
    def fit(self, X, y):
        # Convert data to NumPy array as precaution
        X = np.array(X)
        y = np.array(y)

        # Find mean and standard deviation of X
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Normalize each feature to account for scale
        XNorm = (X - self.mean) / self.std

        # Add a column of 1's to account from w0
        XFinal = np.insert(XNorm, 0, np.ones(X.shape[0]), axis=1)

        # Make a prediction with current W and find loss
        prev_weights = self.w
        new_weights = 0
        reg = self.reg

        # compute loss for the first iteration
        prev_loss = compute_cost(prev_weights, XFinal, y, reg)

        batch_size = self.batch

        # security precaution to avoid improper sizes for batches
        # also takes cake of the special case for m = n
        if batch_size > XFinal.shape[0] or batch_size < 0:
            # defaults to the size of the dataset if outside of range
            batch_size = XFinal.shape[0]

        # Iterate until convergence or maxIterations
        for i in range(self.maxIterations):
            mini_batches = iterate_minibatches(XFinal, y, batch_size, True)
            # iterate for mini-batch gradient descent
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                delta_w = self.eta * calculate_cost_gradient(prev_weights, X_mini, y_mini, reg)
                # Calculate the new W and loss
                new_weights = prev_weights - delta_w
                new_loss = compute_cost(new_weights, X_mini, y_mini, reg)

                # Check for convergence
                if abs(prev_loss - new_loss) < self.epsilon:
                    break

                # Assign the calculated weights, predictions, and loss as old
                prev_weights = new_weights
                prev_loss = new_loss

        self.w = new_weights
    
    # Predict the value of new dataPoints
    def predict(self, X):
        XNorm = (X - self.mean) / self.std
        XFinal = np.insert(XNorm, 0, np.ones(X.shape[0]), axis=1)
        y_pred = []
        for i in np.arange(XFinal.shape[0]):
            # converts to either 1 or -1
            y_hat = np.sign(np.dot(self.w, XFinal[i]))
            y_pred.append(y_hat)

        return np.array(y_pred)
