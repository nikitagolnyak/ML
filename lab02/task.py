from math import sqrt
from random import randrange


def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0] * len(train[0])
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
    return coef


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def dataset_minmax(dataset):
    minmax = []
    size = len(dataset[0])
    for i in range(size):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            deviation = (minmax[i][1] - minmax[i][0])
            if deviation == 0:
                deviation = 1
            row[i] = (row[i] - minmax[i][0]) / deviation


if __name__ == '__main__':
    dataset = []
    y = []
    n_folds = 5
    eps = 0.1
    epochs = 30
    with open('Linear/9.txt') as f:
        n_feat = int(f.readline())
        n_observ = int(f.readline())
        for line in f:
            tmp = [int(x) for x in line.split(' ')]
            if len(tmp) > 1:
                dataset.append(tmp)
    minmax = dataset_minmax(dataset)
    normalize(dataset, minmax)
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for i in range(7):
        eps /= 10
        epochs += 10
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = linear_regression_sgd(train_set, test_set, eps, epochs)
            actual = [row[-1] for row in fold]
            rmse = rmse_metric(actual, predicted)
            scores.append(rmse)
        print('Scores: %s Epochs: %s Eps: %s' % (scores, epochs, eps))
        print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))
        scores.clear()
