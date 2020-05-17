import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import impute_methods

# ******************************** CLASSES ***********************************
class kNN(impute_methods):
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None

    def __distance(self, i, j, q=2):
        sum = 0
        m = 0

        for l in range(self.d):
            if self.O[i][l] == 1 and self.O[j][l] == 1:
                sum += (abs(self.dataset[i][l] - self.dataset[j][l]) ** q)
                m += 1

        if m == 0:
            return float('inf')
        else:
            return (sum/m) ** (1/q)

    def impute(self, dataset, k=5, q=2):
        self.dataset = dataset
        self.n, self.d = dataset.shape

        M = []

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        # For each instance with missing values
        for i in range(self.n):
            elem = np.copy(dataset[i])
            miss_indexes = [x for x, val in enumerate(self.O[i]) if not val]

            for miss_index in miss_indexes:
                # Compute distances to all other elements
                distances = []
                for j in range(self.n):
                    if j != i and not math.isnan(dataset[j][miss_index]):
                        # compute distance
                        dist = self.__distance(i, j, q)
                        if dist != float("inf"):
                            distances.append(tuple((dist, j)))
                distances.sort()
                distances = distances[:k]

                # Impute missing values
                sum = 0
                m = len(distances)

                for [_, other] in distances:
                    sum += dataset[other][miss_index]

                if m != 0:
                    elem[miss_index] = sum/m
                else:
                    elem[miss_index] = 0

            M.append(elem)

        return np.array(M)

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

class wNN(impute_methods):
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None

    def __distance(self, i, j, q=2):
        sum = 0
        m = 0

        for l in range(self.d):
            if self.O[i][l] == 1 and self.O[j][l] == 1:
                sum += (abs(self.dataset[i][l] - self.dataset[j][l]) ** q)
                m += 1

        if m == 0:
            return float('inf')
        else:
            return (sum/m) ** (1/q)

    def __weight(self, dist, sum_distances, kernel_type, lambd):
        return self.__kernelize((dist/lambd), kernel_type)/sum_distances

    def __kernelize(self, val, kernel='Gaussian'):
        if kernel == 'Gaussian':
            return (math.exp(-0.5 * (val ** 2)))/(math.sqrt(2 * math.pi))
        elif kernel == 'Tricube':
            return 70/85 * (1 - abs(val) ** 3) ** 3
        else:
            print("Any kernel selected")

    def impute(self, dataset, k=5, q=2, kernel_type='Gaussian', lambd=5):
        self.dataset = dataset
        self.n, self.d = dataset.shape

        M = []

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        # For each instance with missing values
        for i in range(self.n):
            elem = np.copy(dataset[i])
            miss_indexes = [x for x, val in enumerate(self.O[i]) if not val]

            for miss_index in miss_indexes:
                # Compute distances to all other elements
                distances = []
                for j in range(self.n):
                    if j != i and not math.isnan(dataset[j][miss_index]):
                        # compute distance
                        dist = self.__distance(i, j, q)
                        if dist != float("inf"):
                            distances.append(tuple((dist, j)))
                distances.sort()
                distances = distances[:k]

                # Impute missing values
                sum_distances = 0
                for [dist, _] in distances:
                    sum_distances += self.__kernelize((dist/lambd), kernel_type)

                sum = 0
                for [dist, other] in distances:
                    sum += self.__weight(dist, sum_distances, kernel_type, lambd) * dataset[other][miss_index]

                if sum != 0:
                    elem[miss_index] = sum
                else:
                    elem[miss_index] = 0

            M.append(elem)

        return np.array(M)

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

class wNN_correlation(impute_methods):
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None
        self.cov = None

    def __C(self, r, m=5, c=0.5, type='Power'):
        if type == 'Power':
            return abs(r) ** m
        else:
            if abs(r) > c:
                return abs(r)/(1 - c) - c/(1 - c)
            else:
                return 0

    def __distance(self, i, j, index, q=2):
        sum = 0
        m = 0

        for l in range(self.d):
            if self.O[i][l] == 1 and self.O[j][l] == 1:
                sum += (abs(self.dataset[i][l] - self.dataset[j][l]) ** q) * self.__C(self.cov[l][index])
                m += 1

        if m == 0:
            return float('inf')
        else:
            return (sum/m) ** (1/q)

    def __weight(self, dist, sum_distances, kernel_type, lambd):
        return self.__kernelize((dist/lambd), kernel_type)/sum_distances

    def __kernelize(self, val, kernel='Gaussian'):
        if kernel == 'Gaussian':
            return (math.exp(-0.5 * (val ** 2)))/(math.sqrt(2 * math.pi))
        elif kernel == 'Tricube':
            return 70/85 * (1 - abs(val) ** 3) ** 3
        else:
            print("Any kernel selected")

    def impute(self, real_dataset, dataset, k=5, q=2, kernel_type='Gaussian', lambd=5):
        self.dataset = dataset
        self.n, self.d = dataset.shape
        self.cov = np.cov(real_dataset.T)

        M = []

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        # For each instance with missing values
        for i in range(self.n):
            elem = np.copy(dataset[i])
            miss_indexes = [x for x, val in enumerate(self.O[i]) if not val]

            for miss_index in miss_indexes:
                # Compute distances to all other elements
                distances = []
                for j in range(self.n):
                    if j != i and not math.isnan(dataset[j][miss_index]):
                        # compute distance
                        dist = self.__distance(i, j, miss_index, q)
                        if dist != float("inf"):
                            distances.append(tuple((dist, j)))
                distances.sort()
                distances = distances[:k]

                # Impute missing values
                sum_distances = 0
                for [dist, _] in distances:
                    sum_distances += self.__kernelize((dist/lambd), kernel_type)

                sum = 0
                for [dist, other] in distances:
                    sum += self.__weight(dist, sum_distances, kernel_type, lambd) * dataset[other][miss_index]

                if sum != 0:
                    elem[miss_index] = sum
                else:
                    elem[miss_index] = 0

            M.append(elem)

        return np.array(M)

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)
