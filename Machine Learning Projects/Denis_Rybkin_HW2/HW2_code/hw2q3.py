from sklearn.linear_model import LogisticRegression
from MultiGaussClassify import MultiGaussClassify
from my_cross_val import my_cross_val
from datasets import prepare_boston50, prepare_boston25, prepare_digits
import numpy as np


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
    Boston50_X, Boston50_y, Boston50_k, Boston50_d = prepare_boston50()
    Boston25_X, Boston25_y, Boston25_k, Boston25_d = prepare_boston25()
    Digits_X, Digits_y, Digits_k, Digits_d = prepare_digits()

    default_order = [
        ('MultiGaussClassify with full covariance matrix', 'Boston50'),
        ('MultiGaussClassify with full covariance matrix', 'Boston25'),
        ('MultiGaussClassify with full covariance matrix', 'Digits'),
        ('MultiGaussClassify with diagonal covariance matrix', 'Boston50'),
        ('MultiGaussClassify with diagonal covariance matrix', 'Boston25'),
        ('MultiGaussClassify with diagonal covariance matrix', 'Digits'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston25'),
        ('LogisticRegression', 'Digits')
    ]

    methods = {
        ('MultiGaussClassify with full covariance matrix', 'Boston50'):
        (MultiGaussClassify(Boston50_k, Boston50_d), Boston50_X, Boston50_y),
        ('MultiGaussClassify with full covariance matrix', 'Boston25'):
        (MultiGaussClassify(Boston25_k, Boston25_d), Boston25_X, Boston25_y),
        ('MultiGaussClassify with full covariance matrix', 'Digits'):
        (MultiGaussClassify(Digits_k, Digits_d), Digits_X, Digits_y),
        ('MultiGaussClassify with diagonal covariance matrix', 'Boston50'):
        (MultiGaussClassify(Boston50_k, Boston50_d, True), Boston50_X, Boston50_y),
        ('MultiGaussClassify with diagonal covariance matrix', 'Boston25'):
        (MultiGaussClassify(Boston25_k, Boston25_d, True), Boston25_X, Boston25_y),
        ('MultiGaussClassify with diagonal covariance matrix', 'Digits'):
        (MultiGaussClassify(Digits_k, Digits_d, True), Digits_X, Digits_y),
        ('LogisticRegression', 'Boston50'):
        (LogisticRegression(), Boston50_X, Boston50_y),
        ('LogisticRegression', 'Boston25'):
        (LogisticRegression(), Boston25_X, Boston25_y),
        ('LogisticRegression', 'Digits'):
        (LogisticRegression(), Digits_X, Digits_y)
    }

    for key in default_order:
        name, dataset = key
        method, X, y = methods[key]
        # Using my implementation of cross validation instead of the built-in one
        scores = my_cross_val(method, X, y, 5)
        my_pretty_print(name, dataset, scores)


if __name__ == '__main__':
    # Run q3 whenever we run the program
    q3()
