from sklearn.metrics import confusion_matrix

from cf.C import get_distance_func, get_kernel_func
from cf.Task03 import knn
from lab01.one_hot import distances
from lab01.task import kernels, a_h, Combination, f_measure, a_k


def use_a_h_normal(loo, df, X, y, labels):
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
                    y_pred = a_h(X_train, y_train, X_test, kerfunc, disfunc, h)
                    rounded = min(labels, key=lambda x: abs(x - y_pred)).tolist()
                    preds.append(rounded[0] * 2)
                    flatten = y_test.flatten().tolist()
                    actuals.append(flatten[0] * 2)
                matrix = confusion_matrix(actuals, preds)
                combinations.append(Combination(distance, kernel, f_measure(matrix), h))
                actuals.clear()
                preds.clear()
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)


def use_a_k_normal(loo, df, X, y, labels):
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
                    X_test = X_test.flatten()
                    y_pred = a_k(X_train, y_train, X_test, kerfunc, disfunc, knn(X, X_test, k, disfunc))
                    rounded = min(labels, key=lambda x: abs(x - y_pred)).tolist()
                    preds.append(rounded[0] * 2)
                    flatten = y_test.flatten().tolist()
                    actuals.append(flatten[0] * 2)
                matrix = confusion_matrix(actuals, preds)
                combinations.append(Combination(distance, kernel, f_measure(matrix), k))
                actuals.clear()
                preds.clear()
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)
