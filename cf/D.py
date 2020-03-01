import math
from random import randint

import numpy as np
from numpy import array
from numpy.linalg import pinv


def svd_LeastSquares(A, b):
    U, s, V = np.linalg.svd(A)

    r = [0.0, 0.0]
    r += (1 / s[0]) * (U[:, 0].T * b) * (V[:, 0].T)
    r += (1 / s[1]) * (U[:, 1].T * b) * (V[:, 1].T)

    return r


def predict(row, coefficients):
    y_pred = coefficients[0]
    for i in range(len(row) - 1):
        y_pred += coefficients[i + 1] * row[i]
    return y_pred


def stochastic_gradient_step_v(row, coef, l_rate):
    y_pred = predict(row, coef)
    error = y_pred - row[-1]
    coef[0] = coef[0] - l_rate * error
    for i in range(len(row) - 1):
        coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
    return coef


def coefficients_sgd(train, l_rate, n_epoch):
    coef = [1.0] * len(train[0])
    iter_num = 0
    while iter_num < n_epoch:
        random_ind = randint(0, len(train) - 1)
        new_w = stochastic_gradient_step_v(dataset[random_ind], coef, l_rate)
        coef = new_w
        iter_num += 1
    return coef


def dataset_minmax(dataset):
    minmax = []
    size = len(dataset[0])
    for i in range(size):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def minmax_scale(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            diff = (minmax[i][1] - minmax[i][0])
            if diff == 0:
                diff = 1
            row[i] = (row[i] - minmax[i][0]) / diff


def normalize_back(coef, dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            diff = (minmax[i][1] - minmax[i][0])
            if diff == 0:
                diff = 1
            coef[i] *= row[i] * diff + minmax[i][0]
    return coef


def std_back(coef, dataset, params):
    for row in dataset:
        for i in range(len(row)):
            mean = params[i][0]
            deviation = params[i][1]
            coef[i] *= row[i] * deviation + mean
    return coef


def standardization(dataset, params):
    for row in dataset:
        for i in range(len(row)):
            mean = params[i][0]
            deviation = params[i][1]
            row[i] = (row[i] - mean) / deviation


def compute_mean_deviation(dataset):
    params = []
    size = len(dataset[0])
    for i in range(size):
        col_values = [row[i] for row in dataset]
        mean = sum(col_values) / len(col_values)
        sm = 0
        for x in col_values:
            sm += math.pow(x - mean, 2)
        deviation = math.sqrt(sm / len(col_values))
        params.append([mean, deviation])
    return params


if __name__ == '__main__':
    dataset = []
    eps = 0.001
    epochs = 1000
    tmp = str(input()).split()
    n_observ, n_feat = int(tmp[0]), int(tmp[1])
    for i in range(n_observ):
        line = str(input())
        tmp = [int(x) for x in line.split(' ')]
        dataset.append(tmp)
    params = compute_mean_deviation(dataset)
    standardization(dataset, params)
    coef = coefficients_sgd(dataset, eps, epochs)
    # std_back(coef, dataset, params)
    # minmax = dataset_minmax(dataset)
    # minmax_scale(dataset, minmax)
    # coef = normalize_back(coef, dataset, minmax)
    data = array(dataset)
    X, y = data[:, 0], data[:, 1]
    X = X.reshape((len(X), 1))
    print(pinv(X).dot(y))
    A = np.matrix([[0, 1], [1, 1], [2, 1], [3, 1],
                   [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1],
                   [10, 1], [11, 1], [12, 1], [13, 1], [14, 1]])
    b = np.matrix([[2.2], [2.2], [1], [3], [3], [4], [3], [6],
                   [6], [7], [11], [12], [14], [10], [11]])
    print(svd_LeastSquares(A, b))
    print(coef)
    # i = len(coef)
    # while i > 0:
    #     print(coef[i - 1])
    #     i -= 1
