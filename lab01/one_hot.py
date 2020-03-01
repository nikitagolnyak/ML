import numpy as np
from sklearn.metrics import confusion_matrix

from cf.Task03 import get_distance_func, get_kernel_func, knn
from lab01.task import Combination, f_measure, distances, kernels, a_k, a_h


def use_a_k_one_hot(loo, df, X, y):
    preds = []
    actuals = []
    combinations = []
    k = 0
    for i in range(10):
        k += 1
        for distance in distances:
            disfunc = get_distance_func(distance)
            for kernel in kernels:
                kerfunc = get_kernel_func(kernel)
                for train_index, test_index in loo.split(df):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    y_preds = []
                    X_test = X_test.flatten()
                    for col in zip(*y_train):
                        y_preds.append(a_k(X_train, col, X_test, kerfunc, disfunc, knn(X, X_test, k, disfunc)))
                    y_pred = y_preds.index(max(y_preds))
                    preds.append(y_pred)
                    actual = np.argmax(y_test)
                    actuals.append(actual)
                matrix = confusion_matrix(actuals, preds)
                combinations.append(Combination(distance, kernel, f_measure(matrix), k))
                actuals.clear()
                preds.clear()
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)


def use_a_h_one_hot(loo, df, X, y):
    preds = []
    actuals = []
    combinations = []
    h = 0.1
    for i in range(10):
        h += 0.1
        for distance in distances:
            disfunc = get_distance_func(distance)
            for kernel in kernels:
                kerfunc = get_kernel_func(kernel)
                for train_index, test_index in loo.split(df):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    y_preds = []
                    X_test = X_test.flatten()
                    for col in zip(*y_train):
                        y_preds.append(a_h(X_train, col, X_test, kerfunc, disfunc, h))
                    y_pred = y_preds.index(max(y_preds))
                    preds.append(y_pred)
                    actual = np.argmax(y_test)
                    actuals.append(actual)
                matrix = confusion_matrix(actuals, preds)
                combinations.append(Combination(distance, kernel, f_measure(matrix), h))
                actuals.clear()
                preds.clear()
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)