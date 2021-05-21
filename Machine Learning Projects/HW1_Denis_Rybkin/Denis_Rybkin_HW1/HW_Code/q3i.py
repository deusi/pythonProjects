import numpy as np
import pandas as pd
import my_cross_val

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# for testing purposes
# uncomment 'Test' parts if you want to compare built-in funcs with my implementation
# from sklearn.model_selection import cross_val_score

# stuff for boston datasets creation
label_quality = LabelEncoder()
group_names = ['bad', 'good']

# boston with targets > 50% = 1, else 0
boston50 = load_boston()
bins50 = (2, np.percentile(boston50['target'], 50), np.max(boston50['target']))
boston50['target'] = pd.cut(boston50['target'], bins = bins50, labels= group_names)
boston50['target'] = label_quality.fit_transform(boston50['target'])

# boston with targets > 25% = 1, else 0
boston25 = load_boston()
bins25 = (2, np.percentile(boston25['target'], 25), np.max(boston25['target']))
boston25['target'] = pd.cut(boston25['target'], bins = bins25, labels= group_names)
boston25['target'] = label_quality.fit_transform(boston25['target'])

# digits dataset
digits = load_digits()

# required methods with given parameters
mySVC = SVC(gamma='scale', C=10)
myLinearSVC = LinearSVC(max_iter=2000)
myLR = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)


# Set dataset variables
X_boston50, y_boston50 = boston50.data, boston50.target
X_boston25, y_boston25 = boston25.data, boston25.target
X_digits, y_digits = digits.data, digits.target

count = 1
print("LinearSVC with Boston50:")
b50AccuracyLinearSVC = my_cross_val.my_cross_val(myLinearSVC, X_boston50, y_boston50, 10)
for item in b50AccuracyLinearSVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b50AccuracyLinearSVC.mean()))
print('Standard Deviation: ' + str(b50AccuracyLinearSVC.std()))

"""
print("Test:")
b50aAccuracyLinearSVC_test = cross_val_score(myLinearSVC, X_boston50, y_boston50, cv=10)
print(b50aAccuracyLinearSVC_test)
print(1 - b50aAccuracyLinearSVC_test.mean())
print(b50aAccuracyLinearSVC_test.std())
"""
print(' ')

print("LinearSVC with Boston25:")
b25AccuracyLinearSVC = my_cross_val.my_cross_val(myLinearSVC, X_boston25, y_boston25, 10)
for item in b25AccuracyLinearSVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b25AccuracyLinearSVC.mean()))
print('Standard Deviation: ' + str(b25AccuracyLinearSVC.std()))
"""
print("Test:")
b25aAccuracyLinearSVC_test = cross_val_score(myLinearSVC, X_boston25, y_boston25, cv=10)
print(b25aAccuracyLinearSVC_test)
print(1 - b25aAccuracyLinearSVC_test.mean())
print(b25aAccuracyLinearSVC_test.std())
"""
print(' ')

print("LinearSVC with Digits:")
dAccuracyLinearSVC= my_cross_val.my_cross_val(myLinearSVC, X_digits, y_digits, 10)
for item in dAccuracyLinearSVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLinearSVC.mean()))
print('Standard Deviation: ' + str(dAccuracyLinearSVC.std()))
"""
print("Test:")
daAccuracyLinearSVC_test = cross_val_score(myLinearSVC, X_digits, y_digits, cv=10)
print(daAccuracyLinearSVC_test)
print(1 - daAccuracyLinearSVC_test.mean())
print(daAccuracyLinearSVC_test.std())
"""
print(' ')


print("SVC with Boston50:")
b50AccuracySVC = my_cross_val.my_cross_val(mySVC, X_boston50, y_boston50, 10)
for item in b50AccuracySVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b50AccuracySVC.mean()))
print('Standard Deviation: ' + str(b50AccuracySVC.std()))
"""
print("Test:")
b50aAccuracySVC_test = cross_val_score(mySVC, X_boston50, y_boston50, cv=10)
print(b50aAccuracySVC_test)
print(1 - b50aAccuracySVC_test.mean())
print(b50aAccuracySVC_test.std())
"""
print(' ')

print("SVC with Boston25:")
b25AccuracySVC = my_cross_val.my_cross_val(mySVC, X_boston25, y_boston25, 10)
for item in b25AccuracySVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b25AccuracySVC.mean()))
print('Standard Deviation: ' + str(b25AccuracySVC.std()))
"""
print("Test:")
b25aAccuracySVC_test = cross_val_score(mySVC, X_boston25, y_boston25, cv=10)
print(b25aAccuracySVC_test)
print(1 - b25aAccuracySVC_test.mean())
print(b25aAccuracySVC_test.std())
"""
print(' ')

print("SVC with Digits:")
dAccuracySVC = my_cross_val.my_cross_val(mySVC, X_digits, y_digits, 10)
for item in dAccuracySVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracySVC.mean()))
print('Standard Deviation: ' + str(dAccuracySVC.std()))
"""
print("Test:")
dAccuracySVC_test = cross_val_score(mySVC, X_digits, y_digits, cv=10)
print(dAccuracySVC_test)
print(1 - dAccuracySVC_test.mean())
print(dAccuracySVC_test.std())
"""
print(' ')


print("Logistic Regression with Boston50:")
b50AccuracyLR = my_cross_val.my_cross_val(myLR, X_boston50, y_boston50, 10)
for item in b50AccuracyLR:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b50AccuracyLR.mean()))
print('Standard Deviation: ' + str(b50AccuracyLR.std()))
"""
print("Test:")
b50aAccuracyLR_test = cross_val_score(myLR, X_boston50, y_boston50, cv=10)
print(b50aAccuracyLR_test)
print(1 - b50aAccuracyLR_test.mean())
print(b50aAccuracyLR_test.std())
"""
print(' ')

print("Logistic Regression with Boston25:")
b25AccuracyLR = my_cross_val.my_cross_val(myLR, X_boston25, y_boston25, 10)
for item in b25AccuracyLR:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(b25AccuracyLR.mean()))
print('Standard Deviation: ' + str(b25AccuracyLR.std()))
"""
print("Test:")
b25aAccuracyLR_test = cross_val_score(myLR, X_boston25, y_boston25, cv=10)
print(b25aAccuracyLR_test)
print(1 - b25aAccuracyLR_test.mean())
print(b25aAccuracyLR_test.std())
"""
print(' ')

print("Logistic Regression with Digits:")
dAccuracyLR = my_cross_val.my_cross_val(myLR, X_digits, y_digits, 10)
for item in dAccuracyLR:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLR.mean()))
print('Standard Deviation: ' + str(dAccuracyLR.std()))
"""
print("Test:")
dAccuracyLR_test = cross_val_score(myLR, X_digits, y_digits, cv=10)
print(dAccuracyLR_test)
print(1 - dAccuracyLR_test.mean())
print(dAccuracyLR_test.std())
"""
