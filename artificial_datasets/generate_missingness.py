import math
import random
import numpy as np
import pandas as pd

missingness_percentage = 0.9

n = 200
d = 30

assert math.floor(d * missingness_percentage) != 0

str = """
Choose the type of missingness you want to use:

    [1] Same percentage in each element
    [2] Percentage in the whole data set
      """

print(str)
opt = input("Option: ")

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

# Missigness in each sample
if opt == '1':
    for i in range(n):
        L = random.sample(range(d), math.floor(d * missingness_percentage))
        for j in range(len(L)):
            dataset[i][L[j]] = float('nan')

    dataset = np.array(dataset)

# Missigness in the whole dataset
# Check that any instance has all values equal to NaN
elif opt == '2':
    dataset = dataset.flatten()

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)

# data[[np.random.random_integers(0,10000, 100)],:][:, [np.random.random_integers(0,99, 100)]] = np.nan
print(dataset)
