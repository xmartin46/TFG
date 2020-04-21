import numpy as np
import pandas as pd

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
n = 200
d = 30

rho = 0.9   # pairwise correlation

mean = [0]*d    # [0 for _ in range(d)]
cov = []

for i in range(d):
    A = []
    for j in range(d):
        A.append(rho**(abs(i - j)))
    cov.append(A)
cov = np.array(cov)

# print(cov)

if is_pos_def(cov):
    dataset = np.random.multivariate_normal(mean, cov, n)
    # print(np.cov(dataset))
else:
    print("Error")
