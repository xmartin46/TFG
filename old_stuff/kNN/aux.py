import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

missingness_percentage = 0.15

n = 100
d = 10

assert math.floor(d * missingness_percentage) != 0

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

# AR(1)
rho = 0.9 # pairwise correlation

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
    print("COV => ", cov)
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

print(dataset)
print()
print(O)
print()
print(datasetttttt)
print()

df=pd.DataFrame(data=dataset[0:,0:], index=[i for i in range(n)], columns=['f'+str(i) for i in range(d)])

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

def C(r, m=2):
    return abs(r) ** m
    # c = 0.5
    # if abs(r) > c:
    #     return abs(r)/(1 - c) - c/(1 - c)
    # else:
    #     return 0

def distance_correlation(i, j, index, O, q=2):
    Xi = df.iloc[i].values
    Xj = df.iloc[j].values
    sum = 0
    m = 0

    for k in range(d):
        if O[i][k] == 1 and O[j][k] == 1:
            Ca = C(cov[index][k], 4)
            sum += ((abs(Xi[k] - Xj[k]) * Ca) ** q)
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

def weight(i, j, O, q, sum_distances, total, lambd=1):
    # Expand for different Kernels
    if sum_distances == 0:
        # All distances are infinite => any attribute is valid in both elements i and j
        # Act as if it is not weighted
        return 1/total

    if distance(i, j, O, q) == float("inf"):
        # There exist some element which has attributes valid in both i and j elements
        # This element j has 0 attributes valid when it is valid in element i
        # Return 0
        return 0

    return (distance(i, j, O, q)/lambd)/(sum_distances/lambd)

def weight_correlation(i, j, index, O, q, sum_distances, total, lambd=1):
    # Expand for different Kernels
    if sum_distances == 0:
        # All distances are infinite => any attribute is valid in both elements i and j
        # Act as if it is not weighted
        return 1/total

    if distance_correlation(i, j, index, O, q) == float("inf"):
        # There exist some element which has attributes valid in both i and j elements
        # This element j has 0 attributes valid when it is valid in element i
        # Return 0
        return 0

    return (distance_correlation(i, j, index, O, q)/lambd)/(sum_distances/lambd)

def wNN(dataset, O, k=2):
    M = []
    q = 1

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
                    distances.append(tuple((dist, j)))
            distances.sort()
            distances = distances[:k]

            # impute missing values
            sum_distances = 0
            for [_, other] in distances:
                val = distance(i, other, O, q)
                if val != float('inf'):
                    sum_distances += val

            sum = 0
            for [_, other] in distances:
                sum += weight(i, other, O, q, sum_distances, len(distances)) * df.iloc[other].values[index]

            if sum != 0:
                elem[index] = sum
            else:
                elem[index] = float('nan')

        M.append(elem)

    return np.array(M)

def wNNCorrelation(dataset, O, k=2):
    M = []
    q = 1

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
                    dist = distance_correlation(i, j, index, O, q)
                    distances.append(tuple((dist, j)))
            distances = list(filter(lambda x: x[1] != float("inf"), distances))
            distances.sort()
            distances = distances[:k]

            # impute missing values
            sum_distances = 0
            for [_, other] in distances:
                val = distance_correlation(i, other, index, O, q)
                if val != float('inf'):
                    sum_distances += val

            sum = 0
            for [_, other] in distances:
                sum += weight_correlation(i, other, index, O, q, sum_distances, len(distances)) * df.iloc[other].values[index]

            if sum != 0:
                elem[index] = sum
            else:
                elem[index] = float('nan')

        M.append(elem)

    return np.array(M)


# basicNN(df, O, 2)
M = wNNCorrelation(df, O, 100)
print(M)


print()
print(abs(datasetttttt - M))
print()
difference = np.sum(abs(datasetttttt - M) ** 2)/np.count_nonzero(datasetttttt - M)
print(difference)
