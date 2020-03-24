import numpy as np
import pandas as pd
import random

n = 200
d = 30

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


str = """
Choose the type of randomness you want to use:

    [1] Random list
    [2] Multivariate normal distribution
      """

print(str)
opt = input("Option: ")


if opt == '1':
    dataset = [random_list(d) for _ in range(n)]
    dataset = np.array(dataset)
elif opt == '2':
    mean = random_list(d, secure=True)    # [0 for _ in range(d)]
    cov = np.random.rand(d, d)

    dataset = np.random.multivariate_normal(mean, cov, n).T

print(dataset)
print()
print(np.cov(dataset))
