import random
from random import randint


def predict(row, W):
    y_pred = 0
    for i in range(len(row)):
        y_pred += W[i] * row[i]
    return y_pred


def stochastic_gradient_step_v(row, y_actual, W, Q, lam):
    y_pred = predict(row, W)
    error = y_pred - y_actual
    Q_new = lam * error + (1 - lam) * Q
    if abs(Q_new - Q) < 0.000001:
        return W, Q, True
    l_rate = 0.0
    W_cur = [0] * len(W)
    dx = 0
    for i in range(len(W)):
        W_cur[i] = row[i] * error
        dx += row[i] * W_cur[i]
    if dx != 0:
        if error / dx > l_rate:
            l_rate = error / dx
    if l_rate == 0:
        return W, Q_new, False
    for j in range(len(W)):
        W[j] = W[j] - l_rate * W_cur[j]
    return W, Q, False


def init_loss(X, Y, W):
    Q = 0.0
    size = len(X)
    for i in range(size):
        y_pred = predict(X[i], W)
        Q += y_pred - Y[i]
    return Q / size


def compute_coefficients(X, Y, n_epoch, W):
    iter_num = 0
    lam = 1 / len(X)
    Q = init_loss(X, Y, W)
    while iter_num < n_epoch:
        random_ind = randint(0, len(X) - 1)
        W, Q, stop = stochastic_gradient_step_v(X[random_ind], Y[random_ind], W, Q, lam)
        if stop: return W
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
