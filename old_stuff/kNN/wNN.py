import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

missingness_percentage = 0.25

n = 200
d = 5

assert math.floor(d * missingness_percentage) != 0

def random_list(n, secure=True):
    random_floats = []
    if secure:
        crypto = random.SystemRandom()
        random_float = crypto.random
    else:
        random_float = random.random
    for _ in range(n):
        random_floats.append(random_float())
    return random_floats

dataset = [random_list(d) for _ in range(n)]
dataset = np.array(dataset)

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

datasetttttt = np.array(np.split(np.array(datasetttttt), n))

# dataset = np.array([[100, 2, np.nan], [np.nan, np.nan, 3], [np.nan, 6, 5], [8, np.nan, np.nan]])

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

df=pd.DataFrame(data=dataset[0:,0:], index=[i for i in range(dataset.shape[0])], columns=['f'+str(i) for i in range(dataset.shape[1])])

def distance(i, j, O, q=2):
    Xi = df.iloc[i].values
    Xj = df.iloc[j].values
    sum = 0
    m = 0

    for k in range(d):
        if O[i][k] == 1 and O[i][k] == O[j][k]:
            sum += (abs(Xi[k] - Xj[k]) ** q)
            m += 1

    if m == 0:
        return float('inf')
    else:
        return (sum/m) ** (1/q)

M = []
def basicNN(dataset, O, k=2):
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
                    # dist = distance(elem, df.iloc[j].values)
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


basicNN(df, O, 2)
M = np.array(M)
print(M)


print()
print(datasetttttt - M)
