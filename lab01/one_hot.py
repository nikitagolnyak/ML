import numpy as np
from sklearn.metrics import confusion_matrix

from cf.C import get_distance_func, get_kernel_func
from cf.Task03 import knn
from lab01.task import Combination, f_measure, distances, kernels, a_k, a_h
import matplotlib.pyplot as plt

def use_a_k_one_hot(loo, df, X, y):
    preds = []
    actuals = []
    combinations = []
    k = 0
    kernel = 'cosine'
    distance = 'manhattan'
    for i in range(20):
        # for distance in distances:
        #     disfunc = get_distance_func(distance)
        #     for kernel in kernels:
        #         kerfunc = get_kernel_func(kernel)
        for train_index, test_index in loo.split(df):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_preds = []
            X_test = X_test.flatten()
            for col in zip(*y_train):
                y_preds.append(a_k(X_train, col, X_test, get_kernel_func(kernel), get_distance_func(distance), knn(X, X_test, k, get_distance_func(distance))))
            y_pred = y_preds.index(max(y_preds))
            preds.append(y_pred)
            actual = np.argmax(y_test)
            actuals.append(actual)
        matrix = confusion_matrix(actuals, preds)
        combinations.append(Combination(distance, kernel, f_measure(matrix), k))
        actuals.clear()
        preds.clear()
        k += 1

    # combinations.sort(key=lambda x: x.measure, reverse=True)
    # for combination in combinations:
    #     print(combination)

    x = [combination.k for combination in combinations]
    y = [combination.measure for combination in combinations]
    plt.plot(x, y)
    plt.ylabel('F-measure')
    plt.xlabel('k')
    plt.show()


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
