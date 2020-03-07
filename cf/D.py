import random
from random import randint


def predict(row, W):
    y_pred = 0
    for i in range(len(row)):
        y_pred += W[i] * row[i]
    return y_pred


# def stochastic_gradient_step_v(row, y_actual, W):
#     y_pred = predict(row, W)
#     error = y_pred - y_actual
#     gr = [0] * len(W)
#     dx = 0.0
#     h = 0.0
#     for i in range(len(row)):
#         gr_value = row[i] * error * 2
#         gr[i] += gr_value
#         dx += row[i] * gr_value
#         # W[i] = W[i] - l_rate * error * row[i]
#     if dx != 0:
#         curr_h = error / dx
#         if h < curr_h:
#             h = curr_h
#     if h == 0:
#         return W
#     for i in range(len(W)):
#         W[i] = W[i] - h * gr[i]
#     return W


def stochastic_gradient_step_v(row, y_actual, W):
    y_pred = predict(row, W)
    gr = [0] * len(W)
    h = 0.0
    value = y_pred - y_actual
    gr_value = value * 2
    dx = 0.0
    for j in range(len(row)):
        j_gr_value = row[j] * gr_value
        gr[j] += j_gr_value
        dx += row[j] * j_gr_value
    if dx != 0:
        curr_h = value / dx
        if h < curr_h:
            h = curr_h
    if h == 0:
        return W
    for i in range(len(W)):
        W[i] = W[i] - h * gr[i]
    return W


def compute_coefficients(X, Y, n_epoch, W):
    iter_num = 0
    alpha = 1 / len(X)
    while iter_num < n_epoch:
        random_ind = randint(0, len(X) - 1)
        W = stochastic_gradient_step_v(X[random_ind], Y[random_ind], W)
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
