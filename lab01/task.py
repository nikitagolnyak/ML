import pandas as pd
from sklearn.model_selection import LeaveOneOut

from lab01.normal import *
from lab01.one_hot import *

kernels = ["uniform", "gaussian", "triangular",
           "epanechnikov", "quartic", "triweight",
           "tricube", "cosine", "sigmoid"]

distances = ["manhattan", "euclidean", "chebyshev"]


class Combination:
    def __init__(self, distance, kernel, measure, h):
        self.distance = distance
        self.kernel = kernel
        self.measure = measure
        self.h = h

    def __str__(self):
        return self.distance + ' ' + self.kernel + ' ' \
               + str(self.measure) + " k " + str(self.h)


def a_h(X, Y, q, kernel, distance, h):
    numerator = 0
    denominator = 0
    if h == 0:
        h = 0.0001
    size = len(X)
    # Y = Y.flatten()
    q = q.flatten()
    for i in range(size):
        numerator = numerator + Y[i] * kernel(distance(q, X[i].flatten()) / h)
        denominator = denominator + kernel(distance(q, X[i].flatten()) / h)
    if denominator == 0:
        alpha = 0
        return alpha
    else:
        alpha = numerator / denominator
        return alpha


def a_k(X, Y, q, kernel, distance, neighbours):
    numerator = 0
    denominator = 0
    for i in range(len(X)):
        numerator = numerator + Y[i] * kernel(distance(q, X[i]) / neighbours)
        denominator = denominator + kernel(distance(q, X[i]) / neighbours)
    if denominator == 0:
        alpha = 0
        return alpha
    else:
        alpha = numerator / denominator
        return alpha


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def f_measure(matrix):
    n_class = len(matrix)
    t = []
    c = [0] * n_class
    p = []
    for i in range(n_class):
        arr = matrix[i]
        cur_sum = 0
        for j in range(n_class):
            cur = int(arr[j])
            cur_sum += cur
            if i == j:
                t.append(cur)
            if j == n_class - 1:
                p.append(cur_sum)
            c[j] += cur
    precs = 0
    all = sum(p)
    micro = 0
    for i in range(n_class):
        if p[i] == 0:
            recall = 0
        else:
            recall = t[i] / p[i]
            precs += (t[i] * c[i]) / p[i]
        if c[i] == 0:
            precision = 0
        else:
            precision = t[i] / c[i]
        if recall + precision == 0:
            fc = 0
        else:
            fc = (2.0 * recall * precision) / (recall + precision)
        micro += (c[i] * fc) / all
    w_recall = sum(t) / all
    w_prec = precs / all
    return 2.0 * (w_prec * w_recall) / (w_prec + w_recall)


if __name__ == '__main__':
    print("Use one hot?")
    ans = str(input())
    df = pd.read_csv('data.csv', sep=',')
    loo = LeaveOneOut()
    if ans == "y":
        loo.get_n_splits(df)
        one_hot = pd.get_dummies(df['class'])
        df = df.drop('class', axis=1)
        df = df.join(one_hot)
        df = normalize(df).to_numpy()
        X, y = np.split(df, [-3], axis=1)
        # use_a_k_one_hot(loo, df, X, y)
        use_a_h_one_hot(loo, df, X, y)
    else:
        df = normalize(df).to_numpy()
        X, y = np.split(df, [-1], axis=1)
        labels = np.unique(y, axis=1)
        use_a_k_normal(loo, df, X, y, labels)
        # use_a_h_normal(loo, df, X, y, labels)
