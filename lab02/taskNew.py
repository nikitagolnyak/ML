import math
import random
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import basinhopping
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def nrmse(predicted, y_test):
    rmse = mean_squared_error(y_test, predicted, squared=True)
    y_max, y_min = np.max(y_test), np.min(y_test)
    diff = y_max - y_min
    return rmse / diff


def mse(y_pred, y_actual):
    sum = 0
    for i in range(len(y_actual)):
        sum += math.pow(y_actual[i] - y_pred[i], 2)
    return sum / len(y_pred)
    # return mean_squared_error(y_pred=y_pred, y_true=y_actual)


def b_mse(w, y, X):
    preds = X.dot(w)
    return mean_squared_error(y, preds)


def genetic():
    w_init = 1 / (2 * (n_feat + 1))
    initial_guess = [random.uniform(-w_init, w_init) for _ in range(n_feat + 1)]
    minimizer_kwargs = {"method": "BFGS", "args": (y_train, X_train)}
    niters = [1, 3, 4, 5, 10, 15, 30, 50, 70, 100]
    scores = []
    for niter in niters:
        print('Predicting at', niter)
        res = basinhopping(b_mse, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=niter, disp=True)
        w = []
        for i in range(n_feat + 1):
            w.append(res.x[i])
        y_pred = X_test.dot(w)
        scores.append(nrmse(y_pred, y_test))
    plt.plot(niters, scores)
    plt.title('Basinhopping 8 dataset')
    plt.ylabel('NRMSE')
    plt.xlabel('iterations')
    plt.show()


def compute_results():
    global f, predicted
    stop = 10
    iter_num = 100
    iter = [7, 10, 12, 30]
    with open('Results.txt', 'w') as f:
        # f.write('Ridge with sag\n')
        # for iter_num in iter:
        #     ridge = Ridge(alpha=10000000.0, solver='sag', max_iter=iter_num)
        #     ridge.fit(X_train, y_train)
        #     predicted = ridge.predict(X_test)
        #     metric = mean_squared_error(y_test, predicted)
        #     f.write('RMSE %s Iter %s\n' % (metric, iter_num))
        # #     iter_num += 100
        # # iter_num = 100
        f.write('Ridge with svd\n')
        iter = [1, 3, 6, 8, 10]
        for iter_num in iter:
            ridge_svd = Ridge(solver='svd', max_iter=iter_num)
            ridge_svd.fit(X_train, y_train)
            predicted = ridge_svd.predict(X_test)
            metric = mean_squared_error(y_test, predicted)
            f.write('RMSE %s Iter %s\n' % (metric, iter_num))
            # iter_num += 100
        # iter_num = 100
        # f.write('Lasso\n')
        # for iter_num in iter:
        #     lasso = Lasso(alpha=0.316, max_iter=iter_num)
        #     lasso.fit(X_train, y_train)
        #     predicted = lasso.predict(X_test)
        #     metric = mean_squared_error(y_test, predicted)
        #     f.write('RMSE %s Iter %s\n' % (metric, iter_num))
        #     # iter_num += 100
        # # iter_num = 100
        # f.write('ElasticNet\n')
        # for iter_num in iter:
        #     clf = ElasticNet(alpha=0.0001, l1_ratio=1.0, max_iter=iter_num)
        #     clf.fit(X_train, y_train)
        #     predicted = clf.predict(X_test)
        #     metric = mean_squared_error(y_test, predicted)
        #     f.write('RMSE %s Iter %s\n' % (metric, iter_num))
        #     # iter_num += 100
    f.close()


def use_sgd_regressor():
    global f, predicted
    metrics = []
    with open('SGDRegressor.txt', 'w') as f:
        f.write('Dataset 1\n')
        f.write('Using L2\n')
        alphas = np.logspace(-3, 7, 200)
        iter_nums = [i for i in range(100, 200)]
        for alpha in alphas:
            clf = SGDRegressor(penalty='l2', alpha=alpha)
            clf.fit(X_train, np.ravel(y_train))
            predicted = clf.predict(X_test)
            metric = nrmse(predicted, y_test)
            f.write('RMSE %s It %s\n' % (metric, alpha))
            metrics.append(metric)
        # plt.plot(iter_nums, metrics)
        # plt.title('SGD with L2-regularization 1 dataset test')
        # plt.ylabel('NRMSE')
        # plt.xlabel('iterations')
        # plt.show()
        f.write('Using L1\n')
        metrics.clear()
        l1_ratios = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for l1_ratio in l1_ratios:
            clf = SGDRegressor(penalty='l1', l1_ratio=l1_ratio)
            clf.fit(X_train, np.ravel(y_train))
            predicted = clf.predict(X_test)
            metric = nrmse(predicted, y_test)
            metrics.append(metric)
            f.write('RMSE %s It %s\n' % (metric, l1_ratio))
        # plt.plot(iter_nums, metrics)
        # plt.title('SGD with L1-regularization 1 dataset test')
        # plt.ylabel('NRMSE')
        # plt.xlabel('iterations')
        # plt.show()
    f.close()


def compare_lib_and_my_svd():
    global predicted
    ridge_svd = Ridge(solver='svd')
    ridge_svd.fit(X_train, y_train)
    predicted1 = ridge_svd.predict(X_test)
    predicted2 = ridge_svd.predict(X_train)
    w = pinv(X_train).dot(y_train)
    predicted = X_test.dot(w)
    train_predicted = X_train.dot(w)
    print('RMSE test: %.5f RMSE train: %.5f' % (nrmse(predicted, y_test), nrmse(train_predicted, y_train)))
    print('lib RMSE test: %.5f RMSE train: %.5f' % (nrmse(predicted1, y_test),
                                                nrmse(predicted2, y_train)))


if __name__ == '__main__':
    l_rate = 0.1
    epochs = 100
    filename = 'Linear/1.txt'
    with open(filename) as f:
        n_feat = int(f.readline())
        n_observ = int(f.readline())
    f.close()
    dataset = np.loadtxt('Linear/1.txt', skiprows=2, max_rows=n_observ)
    dataset = normalize(dataset, axis=1, norm='l1')
    X, Y = np.split(dataset, [-1], axis=1)
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=False)
    # write_alphas()
    # compute_results()
    # use_sgd_regressor()
    # genetic()
    compare_lib_and_my_svd()