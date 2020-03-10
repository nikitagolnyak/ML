import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


class Combination:
    def __init__(self, kernel, measure, degree, coef, gamma, c):
        self.kernel = kernel
        self.measure = measure
        self.degree = degree
        self.coef = coef
        self.gamma = gamma
        self.c = c

    def __str__(self):
        return self.kernel + ' ' + str(self.measure) + " degree " + str(self.degree) + \
               " coef " + str(self.coef) + " gamma " + str(self.gamma) + " c " + str(self.c)


def read_data(dataset, norm):
    X = dataset.drop('class', axis=1).to_numpy()
    y = dataset['class'].to_numpy()
    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
    return X, y


def startified_folds():
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=36851234)
    global c, clf, y_pred
    C_2d_range = [1e-1, 1, 1e1]
    for c in C_2d_range:
        clf = svm.SVC(C=c, kernel='linear')
        for train, test in rskf.split(X, y):
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X_test)
            combinations.append(Combination('linear', f1_score(y_test, y_pred, average='weighted'), 0, 0, 0, c))
    for c in C_2d_range:
        for degree in [1, 2, 3, 4, 5, 6]:
            for coef in [0.0, 0.5, 1.5, 2.0, 2.5]:
                clf = svm.SVC(C=c, kernel='poly', degree=degree, coef0=coef)
                for train, test in rskf.split(X, y):
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    combinations.append(
                        Combination('polynomial', f1_score(y[test], y_pred, average='weighted'), degree, coef, 0, c))
    for c in C_2d_range:
        for gamma in [1e-1, 1, 1e1]:
            clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            for train, test in rskf.split(X, y):
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                combinations.append(Combination('rbf', f1_score(y[test], y_pred, average='weighted'), 0, 0, gamma, c))
    for c in C_2d_range:
        for coef in [0.0, 0.5, 1.5, 2.0, 2.5]:
            clf = svm.SVC(C=c, kernel='sigmoid', coef0=coef)
            for train, test in rskf.split(X, y):
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                combinations.append(
                    Combination('sigmoid', f1_score(y[test], y_pred, average='weighted'), 0, coef, 0, c))
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)


def compute_params():
    global c, clf, y_pred
    C_2d_range = [1e-1, 1, 1e1]
    for c in C_2d_range:
        clf = svm.SVC(C=c, kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        combinations.append(Combination('linear', f1_score(y_test, y_pred, average='weighted'), 0, 0, 0, c))
    for c in C_2d_range:
        for degree in [1, 2, 3, 4, 5, 6]:
            for coef in [0.0, 0.5, 1.5, 2.0, 2.5]:
                clf = svm.SVC(C=c, kernel='poly', degree=degree, coef0=coef)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                combinations.append(
                    Combination('polynomial', f1_score(y_test, y_pred, average='weighted'), degree, coef, 0, c))
    for c in C_2d_range:
        for gamma in [1e-1, 1, 1e1]:
            clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            combinations.append(Combination('rbf', f1_score(y_test, y_pred, average='weighted'), 0, 0, gamma, c))
    for c in C_2d_range:
        for coef in [0.0, 0.5, 1.5, 2.0, 2.5]:
            clf = svm.SVC(C=c, kernel='sigmoid', coef0=coef)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            combinations.append(Combination('sigmoid', f1_score(y_test, y_pred, average='weighted'), 0, coef, 0, c))
    combinations.sort(key=lambda x: x.measure, reverse=True)
    for combination in combinations:
        print(combination)


def geyser_clf():
    global clf
    clf = svm.SVC(kernel='linear', C=10)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='poly', degree=2, coef0=0.5, C=0.1)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='rbf', gamma=10, C=0.1)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='sigmoid', coef0=0.0, C=0.1)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))


def chips_clf():
    global clf
    clf = svm.SVC(kernel='linear', C=10.0)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='poly', degree=2, coef0=0.5, C=1.0)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='rbf', gamma=1, C=10.0)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))
    clf = svm.SVC(kernel='sigmoid', coef0=2.5, C=10.0)
    clf.fit(X, y)
    classifiers.append((clf, clf.kernel))


def rbf_plot(dataset):
    global clf
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    X_2d = X[:, :2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1
    for (k, (clf, kernel)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.title("dataset %s kernel=%s" % (dataset, kernel),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')
        plt.show()


if __name__ == '__main__':
    combinations = []
    classifiers = []
    chips = pd.read_csv("chips.csv")
    X, y = read_data(chips, False)
    # geyser = pd.read_csv("geyser.csv")
    # X, y = read_data(geyser, True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    # compute_params()
    # startified_folds()
    # geyser_clf()
    chips_clf()
    # rbf_plot('geyser')
    for (k, (clf, kernel)) in enumerate(classifiers):
        plt.figure()
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.title(kernel + ' chips')
        plt.show()
