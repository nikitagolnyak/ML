import math


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