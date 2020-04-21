import sys
sys.path.append('..')

import math
import copy
import numpy as np
from scipy.spatial import distance

from methods.ESD import ESD
from methods.utils import *

from sklearn.metrics import mean_squared_error

# ************************ VARIABLES *******************************
missingness_percentage = [0.05, 0.15, 0.3, 0.6]
iterations = 10

model = ESD()

for mis in missingness_percentage:
    print(f"ESD MISSINGNESS PERCENTAGE => {mis}")
    RMSE = 0
    mine = 0

    for it in range(iterations):
        # dataset = generate_dataset_AR(n, d, rho)

        n = 100
        mean = np.array([-0.3, 0.1, 2])
        cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
        dataset = np.random.multivariate_normal(mean, cov, n)

        from sklearn import datasets
        iris = datasets.load_iris()
        dataset = iris.data

        # from auto_mpg import mpg
        # dataset = mpg

        from ecoli import ecoli
        dataset = ecoli

        # from glass import glass
        # dataset = glass

        # from wine import wine
        # dataset = wine

        # dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_missingness_flatten(dataset, mis)

        n, d = dataset_missing.shape

        P = model.estimateDistances(copy.deepcopy(dataset_missing), 1)
        R = realDistances(dataset)

        s = 0
        real = []
        not_real = []
        nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
        for i in range(n):
            for j in range(n):
                if i > j:
                    s += (math.sqrt(P[i][j]) - math.sqrt(R[i][j])) ** 2
                    real.append(math.sqrt(R[i][j]))
                    not_real.append(math.sqrt(P[i][j]))

        count = len(nan_indices)
        RMSE += math.sqrt(s/(count * n - count * (count + 1)/2))
        print(f"RMSE => {RMSE/(it + 1)} ({math.sqrt(s/(count * n - count * (count + 1)/2))})")

        mine += math.sqrt(mean_squared_error(real, not_real))
        print(f"mine => {mine/(it + 1)} ({math.sqrt(mean_squared_error(real, not_real))})")


    print()
    print(f"RMSE => {RMSE/iterations}")
    print(f"mine => {mine/iterations}")
    print()
