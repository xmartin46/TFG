import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

missingness_percentage = 0.05

n = 10
d = 5
neighbors = 10
lambd = 5
samples = 1

kernels = ['Gaussian',
           'Tricube']
kernelType = kernels[0]

rho = 0.9 # pairwise correlation

q = 2
mC = 6

# assert math.floor(d * missingness_percentage) != 0

# FUNCTIONS
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def distance(i, j, O, q=2):
    Xi = df.iloc[i].values
    Xj = df.iloc[j].values
    sum = 0
    m = 0

    for k in range(d):
        if O[i][k] == 1 and O[j][k] == 1:
            sum += (abs(Xi[k] - Xj[k]) ** q)
            m += 1

    if m == 0:
        return float('inf')
    else:
        return (sum/m) ** (1/q)

def weight(dist, sum_distances, total, q, lambd=1):
    # Expand for different Kernels
    if sum_distances == 0:
        # All distances are infinite => any attribute is valid in both elements i and j
        # Act as if it is not weighted
        return 1/total

    # Filtered in the wNN function
    # if dist == float("inf"):
    #     # There exist some element which has attributes valid in both i and j elements
    #     # This element j has 0 attributes valid when it is valid in element i
    #     # Return 0
    #     return 0

    return Kernelize((dist/lambd), kernelType)/(sum_distances/lambd)

def Kernelize(val, kernel='Gaussian'):
    if kernel == 'Gaussian':
        return (math.exp(-0.5 * (val ** 2)))/(math.sqrt(2 * math.pi))
    elif kernel == 'Tricube':
        return 70/85 * (1 - abs(val) ** 3) ** 3
    else:
        print("Any kernel selected")

def wNN(dataset, O, k=2, lambd=5):
    assert lambd != 0

    M = []
    q = 2

    # for each instance with missing values
    for i in range(n):
        elem = np.copy(df.iloc[i].values)
        indexes = [x for x, val in enumerate(O[i]) if not val]

        for index in indexes:
            # Compute distances to all other
            distances = []
            for j in range(n):
                if j != i and not math.isnan(df.iloc[j].values[index]):
                    # compute distance
                    dist = distance(i, j, O, q)
                    if dist != float("inf"):
                        distances.append(tuple((dist, j)))
            distances.sort()
            distances = distances[:k]   # first k neighbors

            # impute missing values
            sum_distances = 0
            for [dist, _] in distances:
                sum_distances += Kernelize((dist/lambd), kernelType)

            sum = 0
            for [dist, other] in distances:
                sum += weight(dist, sum_distances, len(distances), q) * df.iloc[other].values[index]

            if sum != 0:
                elem[index] = sum
            else:
                elem[index] = float('nan')

        M.append(elem)

    return np.array(M)


# ********************
def distance(i, j, O, q=2):
    Xi = df.iloc[i].values
    Xj = df.iloc[j].values
    sum = 0
    m = 0

    for k in range(d):
        if O[i][k] == 1 and O[j][k] == 1:
            sum += (abs(Xi[k] - Xj[k]) ** q)
            m += 1

    if m == 0:
        return float('inf')
    else:
        return (sum/m) ** (1/q)

def basicNN(dataset, O, k=2):
    M = []

    # for each instance with missing values
    for i in range(n):
        elem = np.copy(df.iloc[i].values)
        indexes = [x for x, val in enumerate(O[i]) if not val]

        for index in indexes:
            # Compute distances to all other
            distances = []
            for j in range(n):
                if j != i and not math.isnan(df.iloc[j].values[index]):
                    # compute distance
                    dist = distance(i, j, O, 1)
                    distances.append(tuple((dist, j)))
            distances.sort()
            distances = distances[:k]

            # impute missing values
            sum = 0
            m = 0
            for [_, other] in distances:
                sum += df.iloc[other].values[index]
                m += 1
            if m != 0:
                elem[index] = sum/m
            else:
                elem[index] = float('nan')

        M.append(elem)

    return np.array(M)
# ********************






def C(r, mC=5):
    # c = 0.5
    # if abs(r) > c:
    #     return abs(r)/(1 - c) - c/(1 - c)
    # else:
    #     return 0

    return abs(r) ** mC

def distance_correlation(i, j, index, mC, O, q=2):
    Xi = df.iloc[i].values
    Xj = df.iloc[j].values
    sum = 0
    m = 0

    for k in range(d):
        if O[i][k] == 1 and O[j][k] == 1:
            sum += (abs(Xi[k] - Xj[k]) ** q) * C(cov[index][k], mC)
            m += 1

    if m == 0:
        return float('inf')
    else:
        return (sum/m) ** (1/q)

def weight_correlation(dist, sum_distances, total, q, lambd=1):
    # Expand for different Kernels
    if sum_distances == 0:
        # All distances are infinite => any attribute is valid in both elements i and j
        # Act as if it is not weighted
        return 1/total

    # Filtered in the wNN function
    # if dist == float("inf"):
    #     # There exist some element which has attributes valid in both i and j elements
    #     # This element j has 0 attributes valid when it is valid in element i
    #     # Return 0
    #     return 0

    return Kernelize((dist/lambd), kernelType)/(sum_distances/lambd)

def wNN_correlation(dataset, O, k=2, lambd=5):
    assert lambd != 0

    M = []

    # for each instance with missing values
    for i in range(n):
        elem = np.copy(df.iloc[i].values)
        indexes = [x for x, val in enumerate(O[i]) if not val]

        for index in indexes:
            # Compute distances to all other
            distances = []
            for j in range(n):
                if j != i and not math.isnan(df.iloc[j].values[index]):
                    # compute distance
                    dist = distance_correlation(i, j, index, mC, O, q)
                    if dist != float("inf"):
                        distances.append(tuple((dist, j)))
            distances.sort()
            distances = distances[:k]   # first k neighbors

            # impute missing values
            sum_distances = 0
            for [dist, _] in distances:
                sum_distances += Kernelize((dist/lambd), kernelType)

            sum = 0
            for [dist, other] in distances:
                sum += weight_correlation(dist, sum_distances, len(distances), q) * df.iloc[other].values[index]

            if sum != 0:
                elem[index] = sum
            else:
                elem[index] = float('nan')

        M.append(elem)

    return np.array(M)











MSIE = 0
MAIE = 0

MSIE_w = 0
MAIE_w = 0

MSIE_c = 0
MAIE_c = 0
for _ in tqdm(range(samples)):
    mean = [0]*d    # [0 for _ in range(d)]
    cov = []

    for i in range(d):
        A = []
        for j in range(d):
            A.append(rho**(abs(i - j)))
        cov.append(A)
    cov = np.array(cov)

    if is_pos_def(cov):
        dataset = np.random.multivariate_normal(mean, cov, n)
        cov = np.cov(dataset.T)
        # print(np.cov(dataset))
    else:
        print("Error")











    n, d = dataset.shape

    # Missigness in the whole dataset
    dataset = dataset.flatten()
    datasetttttt = np.copy(dataset.flatten())

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = np.array(dataset)

    # dataset = np.array([[100, np.nan, np.nan, np.nan, np.nan], [np.nan, 2, np.nan, np.nan, np.nan]])
    # datasetttttt = np.copy(dataset.flatten())
    # cov = np.cov(dataset.T)
    # n, d = dataset.shape
    datasetttttt = np.array(np.split(np.array(datasetttttt), n))

    # Observation matrix
    # True      , if attribute is observed
    # False     , otherwise
    O = ~np.isnan(dataset)

    # print(dataset)
    # print()
    # print(O)
    # print()
    # print(datasetttttt)
    # print()

    df=pd.DataFrame(data=dataset[0:,0:], index=[i for i in range(n)], columns=['f'+str(i) for i in range(d)])

    for i in range(n):
        elem = df.iloc[i].values
        found = False
        for j in range(d):
            if not math.isnan(elem[j]):
                found = True
        if not found:
            print("UN SENSE CAP ELEMENT")
            print(elem)
            while 1:
                a = 0

    M = wNN_correlation(df, O, neighbors, lambd)
    # print(M)
    #
    # print()
    # print(abs(datasetttttt - M))
    # print()
    MSIE_c += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    # print("MSIE => ", MSIE)
    MAIE_c += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)
    # print("MAIE => ", MAIE)

    M = wNN(dataset, O, neighbors, lambd)
    MSIE_w += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    MAIE_w += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)

    M = basicNN(dataset, O, neighbors)
    MSIE += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    MAIE += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)

print("MSIE => ", MSIE/samples)
print("MAIE => ", MAIE/samples)

print("MSIE wNN => ", MSIE_w/samples)
print("MAIE wNN => ", MAIE_w/samples)

print("MSIE wNN CORRELATION => ", MSIE_c/samples)
print("MAIE wNN CORRELATION => ", MAIE_c/samples)






MSIE = 0
MAIE = 0

MSIE_w = 0
MAIE_w = 0

MSIE_c = 0
MAIE_c = 0
for _ in tqdm(range(samples)):
    predictors_per_block = 10
    assert d%predictors_per_block == 0

    rho_w = 0.9 # pairwise correlation for components in the same block (within correlation)
    rho_b = 0.1 # pairwise correlation for components in different blocks (between correlation)

    mean = [0]*d    # [0 for _ in range(d)]
    cov = []

    for i in range(d):
        A = []
        for j in range(int(d/predictors_per_block)):
            if math.floor(i/predictors_per_block) == j:
                A.append([rho_w] * predictors_per_block)
            else:
                A.append([rho_b] * predictors_per_block)
        A = np.array(A)
        cov.append(A.flatten())

    cov = np.array(cov)
    # print(cov)
    # print()
    # print()

    dataset = np.random.multivariate_normal(mean, cov, n)
    cov = np.cov(dataset.T)

    n, d = dataset.shape

    # Missigness in the whole dataset
    dataset = dataset.flatten()
    datasetttttt = np.copy(dataset.flatten())

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = np.array(dataset)

    # dataset = np.array([[100, np.nan, np.nan, np.nan, np.nan], [np.nan, 2, np.nan, np.nan, np.nan]])
    # datasetttttt = np.copy(dataset.flatten())
    # cov = np.cov(dataset.T)
    # n, d = dataset.shape
    datasetttttt = np.array(np.split(np.array(datasetttttt), n))

    # Observation matrix
    # True      , if attribute is observed
    # False     , otherwise
    O = ~np.isnan(dataset)

    # print(dataset)
    # print()
    # print(O)
    # print()
    # print(datasetttttt)
    # print()

    df=pd.DataFrame(data=dataset[0:,0:], index=[i for i in range(n)], columns=['f'+str(i) for i in range(d)])

    for i in range(n):
        elem = df.iloc[i].values
        found = False
        for j in range(d):
            if not math.isnan(elem[j]):
                found = True
        if not found:
            print("UN SENSE CAP ELEMENT")
            print(elem)
            while 1:
                a = 0

    M = wNN_correlation(df, O, neighbors, lambd)
    # print(M)
    #
    # print()
    # print(abs(datasetttttt - M))
    # print()
    MSIE_c += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    # print("MSIE => ", MSIE)
    MAIE_c += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)
    # print("MAIE => ", MAIE)

    M = wNN(dataset, O, neighbors, lambd)
    MSIE_w += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    MAIE_w += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)

    M = basicNN(dataset, O, neighbors)
    MSIE += np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
    MAIE += np.sum(abs(datasetttttt - M))/np.count_nonzero(datasetttttt - M)

print("MSIE => ", MSIE/samples)
print("MAIE => ", MAIE/samples)

print("MSIE wNN => ", MSIE_w/samples)
print("MAIE wNN => ", MAIE_w/samples)

print("MSIE wNN CORRELATION => ", MSIE_c/samples)
print("MAIE wNN CORRELATION => ", MAIE_c/samples)
