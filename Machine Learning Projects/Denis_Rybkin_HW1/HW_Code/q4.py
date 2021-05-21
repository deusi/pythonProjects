import my_cross_val
import quad_proj
import rand_proj

from sklearn.datasets import load_digits

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# for testing purposes
# from sklearn.model_selection import cross_val_score

# digits dataset
digits = load_digits()

# required methods with given parameters
mySVC = SVC(gamma='scale', C=10)
myLinearSVC = LinearSVC(max_iter=2000)
myLR = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)


X_digits, y_digits = digits.data, digits.target

X_1 = rand_proj.rand_proj(X_digits, 32)
X_2 = quad_proj.quad_proj(X_digits)


count = 1
print("LinearSVC with X_1:")
dAccuracyLinearSVC= my_cross_val.my_cross_val(myLinearSVC, X_1, y_digits, 10)
for item in dAccuracyLinearSVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLinearSVC.mean()))
print('Standard Deviation: ' + str(dAccuracyLinearSVC.std()))
"""
print("Test:")
daAccuracyLinearSVC_test = cross_val_score(myLinearSVC, X_tilda, y_digits, cv=10)
print(daAccuracyLinearSVC_test)
print(1 - daAccuracyLinearSVC_test.mean())
print(daAccuracyLinearSVC_test.std())
"""
print(' ')


print("LinearSVC with X_2:")
dAccuracyLinearSVC= my_cross_val.my_cross_val(myLinearSVC, X_2, y_digits, 10)
for item in dAccuracyLinearSVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLinearSVC.mean()))
print('Standard Deviation: ' + str(dAccuracyLinearSVC.std()))
"""
print("Test:")
daAccuracyLinearSVC_test = cross_val_score(myLinearSVC, X_2, y_digits, cv=10)
print(daAccuracyLinearSVC_test)
print(1 - daAccuracyLinearSVC_test.mean())
print(daAccuracyLinearSVC_test.std())
"""
print(' ')


print("SVC with X_1:")
dAccuracySVC = my_cross_val.my_cross_val(mySVC, X_1, y_digits, 10)
for item in dAccuracySVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracySVC.mean()))
print('Standard Deviation: ' + str(dAccuracySVC.std()))
"""
print("Test:")
dAccuracySVC_test = cross_val_score(mySVC, X_tilda, y_digits, cv=10)
print(dAccuracySVC_test)
print(1 - dAccuracySVC_test.mean())
print(dAccuracySVC_test.std())
"""
print(' ')


print("SVC with X_2:")
dAccuracySVC = my_cross_val.my_cross_val(mySVC, X_2, y_digits, 10)
for item in dAccuracySVC:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracySVC.mean()))
print('Standard Deviation: ' + str(dAccuracySVC.std()))
"""
print("Test:")
dAccuracySVC_test = cross_val_score(mySVC, X_2, y_digits, cv=10)
print(dAccuracySVC_test)
print(1 - dAccuracySVC_test.mean())
print(dAccuracySVC_test.std())
"""
print(' ')


print("Logistic Regression with X_1:")
dAccuracyLR = my_cross_val.my_cross_val(myLR, X_1, y_digits, 10)
for item in dAccuracyLR:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLR.mean()))
print('Standard Deviation: ' + str(dAccuracyLR.std()))
"""
print("Test:")
dAccuracyLR_test = cross_val_score(myLR, X_tilda, y_digits, cv=10)
print(dAccuracyLR_test)
print(1 - dAccuracyLR_test.mean())
print(dAccuracyLR_test.std())
"""
print(' ')


print("Logistic Regression with X_2:")
dAccuracyLR = my_cross_val.my_cross_val(myLR, X_2, y_digits, 10)
for item in dAccuracyLR:
    print('Fold ' + str(count) + ': ' + str(item))
    count += 1
count = 1
print('Mean: ' + str(dAccuracyLR.mean()))
print('Standard Deviation: ' + str(dAccuracyLR.std()))
"""
print("Test:")
dAccuracyLR_test = cross_val_score(myLR, X_2, y_digits, cv=10)
print(dAccuracyLR_test)
print(1 - dAccuracyLR_test.mean())
print(dAccuracyLR_test.std())
"""

