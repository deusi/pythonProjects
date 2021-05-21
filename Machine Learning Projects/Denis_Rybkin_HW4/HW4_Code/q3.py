import numpy as np
from sklearn.linear_model import LogisticRegression
from MySVM2 import MySVM2
from my_cross_val import my_cross_val
from datasets import prepare_boston25, prepare_boston50


def my_pretty_print(name, dataset, scores):
    # Print the test information into the terminal
    errors = np.array(scores)
    errors = 1 - errors
    print('==============')
    print('Method: {}'.format(name))
    print('Dataset: {}'.format(dataset))
    print('Error Rates:')
    i = 0
    for error in errors:
        i += 1
        print('Fold {} {:.4f}'.format(i, error))
    print('Mean: {:.4f}'.format(np.mean(errors)))
    print('Standard Deviation: {:.4f}'.format(np.std(errors)))


def q3():
    # Initialize the values from the datasets
    boston25_x, boston25_y = prepare_boston25()
    boston50_x, boston50_y = prepare_boston50()

    # Number of folds: assignment specifies the value of 5
    k_fold = 5

    default_order = [
        ('MySVM2 with m = 40', 'Boston50'),
        ('MySVM2 with m = 200', 'Boston50'),
        ('MySVM2 with m = n', 'Boston50'),
        ('LogisticRegression', 'Boston50'),
        ('MySVM2 with m = 40', 'Boston25'),
        ('MySVM2 with m = 200', 'Boston25'),
        ('MySVM2 with m = n', 'Boston25'),
        ('LogisticRegression', 'Boston25')
    ]
    methods = {
        ('MySVM2 with m = 40', 'Boston50'):
            (MySVM2(boston50_x.shape[1], 500, 40), boston50_x, boston50_y),
        ('MySVM2 with m = 200', 'Boston50'):
            (MySVM2(boston50_x.shape[1], 500, 200), boston50_x, boston50_y),
        ('MySVM2 with m = n', 'Boston50'):
        # note that we deliberately pass boston50_x.shape[0] to trigger special case that makes batch size m = n
            (MySVM2(boston50_x.shape[1], 500, boston50_x.shape[0]), boston50_x, boston50_y),
        ('LogisticRegression', 'Boston50'):
            (LogisticRegression(), boston50_x, boston50_y),
        ('MySVM2 with m = 40', 'Boston25'):
            (MySVM2(boston25_x.shape[1], 500, 40), boston25_x, boston25_y),
        ('MySVM2 with m = 200', 'Boston25'):
            (MySVM2(boston25_x.shape[1], 500, 200), boston25_x, boston25_y),
        ('MySVM2 with m = n', 'Boston25'):
        # note that we deliberately pass boston25_x.shape[0] to trigger special case that makes batch size m = n
            (MySVM2(boston25_x.shape[1], 500, boston25_x.shape[0]), boston25_x, boston25_y),
        ('LogisticRegression', 'Boston25'):
            (LogisticRegression(), boston25_x, boston25_y)
    }

    for key in default_order:
        name, dataset = key
        method, x, y = methods[key]
        # Using my implementation of cross validation instead of the built-in one
        scores = my_cross_val(method, x, y, k_fold)
        my_pretty_print(name, dataset, scores)
    print('==============')


if __name__ == '__main__':
    # Run q3 whenever we run the program
    q3()
