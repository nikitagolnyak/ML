import math


def knn(X, q, k, distance):
    distances = []
    for i in range(len(X)):
        distances.append(distance(X[i], q))
    distances = [i for i in distances if i > 0]
    del distances[k + 1:]
    return distances[k]


def ManhattanDistance(X, U):
    dist = 0
    for i in range(len(X)):
        dist = dist + math.fabs(X[i] - U[i])
    return dist


def EuclidianDistance(X, U):
    dist = 0
    for i in range(len(X)):
        dist = dist + ((X[i] - U[i]) ** 2)
    return math.sqrt(dist)


def ChebyshevDistance(X, U):
    dist = 0
    for i in range(len(X)):
        if math.fabs(X[i] - U[i]) > dist:
            dist = math.fabs(X[i] - U[i])
    return dist


def a_k(X, Y, q, kernel, distance, k):
    numerator = 0
    denominator = 0
    neighbours = knn(X, q, k, distance)
    if neighbours == 0:
        neighbours = 1
    for i in range(len(X)):
        numerator = numerator + Y[i] * kernel(distance(q, X[i]) / neighbours)
        denominator = denominator + kernel(distance(q, X[i]) / neighbours)
    if denominator == 0:
        alpha = 0
        return alpha
    else:
        alpha = numerator / denominator
        return alpha


def a_h(X, Y, q, kernel, distance, h):
    numerator = 0
    denominator = 0
    if h == 0:
        return math.nan
    for i in range(len(X)):
        numerator = numerator + Y[i] * kernel(distance(q, X[i]) / h)
        denominator = denominator + kernel(distance(q, X[i]) / h)
    if denominator == 0:
        alpha = 0
        return alpha
    else:
        alpha = numerator / denominator
        return alpha


def get_distance_func(name):
    if name == "manhattan":
        return ManhattanDistance
    elif name == "euclidean":
        return EuclidianDistance
    elif name == "chebyshev":
        return ChebyshevDistance


def UniformKernel(u):
    if math.fabs(u) >= 1:
        return 0
    else:
        return 1.0 / 2.0


def TriangularKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return 1 - math.fabs(u)


def EpanechnikovKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return 0.75 * (1 - u * u)


def QuarticKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return (15 / 16) * math.pow(1 - u * u, 2)


def TriweightKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return (35 / 32) * math.pow(1 - u * u, 3)


def TricubeKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return (70 / 81) * math.pow(1 - math.pow(math.fabs(u), 3), 3)


def CosineKernel(u):
    if math.fabs(u) > 1:
        return 0
    else:
        return (math.pi / 4) * math.cos((math.pi / 2) * u)


def get_kernel_func(name):
    if name == "uniform":
        return UniformKernel
    elif name == "gaussian":
        return lambda u: math.exp(-0.5 * u * u) / math.sqrt(math.pi + math.pi)
    elif name == "triangular":
        return TriangularKernel
    elif name == "epanechnikov":
        return EpanechnikovKernel
    elif name == "quartic":
        return QuarticKernel
    elif name == "triweight":
        return TriweightKernel
    elif name == "tricube":
        return TricubeKernel
    elif name == "cosine":
        return CosineKernel
    elif name == "sigmoid":
        return lambda u: (2 / math.pi) * (1 / (math.exp(u) + math.exp(-u)))


if __name__ == '__main__':
    temp = str(input()).split(' ')
    observations = int(temp[0])
    attr_size = int(temp[1])
    X = []
    y = []
    for i in range(observations):
        temp = str(input()).split(' ')
        tempr = []
        for j in range(attr_size):
            tempr.append(int(temp[j]))
        X.append(tempr)
        y.append(int(temp[attr_size]))
    q = []
    temp = str(input()).split(' ')
    for j in range(attr_size):
        q.append(int(temp[j]))
    distance = str(input())
    kernel = str(input())
    window = str(input())
    res = -1
    if window == 'fixed':
        h = int(input())
        res = a_h(X, y, q, get_kernel_func(kernel), get_distance_func(distance), h)
    elif window == 'variable':
        k = int(input())
        res = a_k(X, y, q, get_kernel_func(kernel), get_distance_func(distance), k)
    print(res)
