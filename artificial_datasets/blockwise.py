import math
import numpy as np
import pandas as pd

n = 200
d = 30

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

dataset = np.random.multivariate_normal(mean, cov, n).T
# print(np.around(np.cov(dataset), 1))
