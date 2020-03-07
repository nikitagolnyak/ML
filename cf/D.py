import math
import random
from random import randint


def predict(row, W):
    y_pred = 0
    for i in range(len(row)):
        y_pred += W[i] * row[i]
    return y_pred


def init_loss(X, Y, W):
    Q = 0.0
    size = len(X)
    for i in range(size):
        y_pred = predict(X[i], W)
        Q += y_pred - Y[i]
    return Q / size


def compute_coefficients(X, Y, n_epoch, W):
    iter_num = 0
    lm = 1 / len(X)
    Q = init_loss(X, Y, W)
    l_rate = 0.00001
    prev_error = math.inf
    prev_W = list(W)
    while iter_num < n_epoch:
        random_ind = randint(0, len(X) - 1)
        x, y = X[random_ind], Y[random_ind]
        y_pred = predict(x, prev_W)
        error = y_pred - y
        if prev_error != math.inf:
            if error > prev_error:
                l_rate = l_rate / 0.5
                prev_W = list(W)
            else:
                l_rate = l_rate * 0.03
                W = list(prev_W)
        Q_new = lm * error + (1 - lm) * Q
        if abs(Q_new - Q) < 0.00001:
            return W
        for j in range(len(W)):
            prev_W[j] = prev_W[j] - x[j] * l_rate * error
        prev_error = error
        Q = Q_new
        iter_num += 1
    return W


if __name__ == '__main__':
    epochs = 75000
    tmp = str(input()).split()
    n_observ, n_feat = int(tmp[0]), int(tmp[1])
    X = []
    Y = []
    for i in range(n_observ):
        line = str(input())
        tmp = [int(x) for x in line.split(' ')]
        tmp_x = tmp[:n_feat]
        tmp_x.append(1)
        X.append(tmp_x)
        Y.append(tmp[-1])
    w_init = 1 / (2 * (n_feat + 1))
    W = [random.uniform(-w_init, w_init) for _ in range(n_feat + 1)]
    W = compute_coefficients(X, Y, epochs, W)
    for w in W:
        print(w)
