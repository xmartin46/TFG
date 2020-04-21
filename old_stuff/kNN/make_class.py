import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ******************************** CLASSES ***********************************
class impute_methods:
    def __init__(self):
        pass

    def MSIE(self, real, imputed):
        return np.sum(abs(real - imputed) ** 2)/np.count_nonzero(real - imputed)

    def MAIE(self, real, imputed):
        return np.sum(abs(real - imputed))/np.count_nonzero(real - imputed)

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
                    elem[miss_index] = float('nan')

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
                    elem[miss_index] = float('nan')

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
                    elem[miss_index] = float('nan')

            M.append(elem)

        return np.array(M)

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

# ********************************************************************************

# *************************** FUNCTIONS *****************************
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def generate_dataset_AR(n, d, rho=0.9):
    mean = [0]*d
    cov = []

    for i in range(d):
        A = []
        for j in range(d):
            A.append(rho**(abs(i - j)))
        cov.append(A)
    cov = np.array(cov)

    if is_pos_def(cov):
        dataset = np.random.multivariate_normal(mean, cov, n)
    else:
        print("Error")

    return dataset

def generate_dataset_blockwise(n, d, rho_w, who_b, predictors_per_block = 10):
    assert d%predictors_per_block == 0

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
    dataset = np.random.multivariate_normal(mean, cov, n)

    return dataset

def generate_missingness(dataset, missingness_percentage):
    n, d = dataset.shape

    dataset = dataset.flatten()

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = np.array(dataset)

    return dataset

def verify_dataset(dataset):
    n, d = dataset.shape

    for i in range(n):
        elem = dataset[i]
        found = False
        for j in range(d):
            if not math.isnan(elem[j]):
                found = True
        if not found:
            print("UN SENSE CAP ELEMENT")
            print(elem)
            while 1:
                a = 0
# *******************************************************************

# *************************** VARIABLES *****************************
# REPETITIONS
samples = 1

# GENERATE DATASETS
n = 100
d = 15
#   AR
rho = 0.9
#   BW
rho_w = 0.9
rho_b = 0.1
predictors_per_block = 5

# MISSINGNESS
missingness_percentage = 0.05

# NEAREST NEIGHBOR
neighbors = 10
q = 2
lambd = 1
mC = 6
kernels = ['Gaussian',
           'Tricube']
kernel_type = kernels[0]
# ******************************************************************

MSIE_AR_kNN = 0
MAIE_AR_kNN = 0

MSIE_BW_kNN = 0
MAIE_BW_kNN = 0


MSIE_AR_wNN = 0
MAIE_AR_wNN = 0

MSIE_BW_wNN = 0
MAIE_BW_wNN = 0


MSIE_AR_wNNC = 0
MAIE_AR_wNNC = 0

MSIE_BW_wNNC = 0
MAIE_BW_wNNC = 0


kNN = kNN()
wNN = wNN()
wNNC = wNN_correlation()

for _ in tqdm(range(samples)):
    datasetAR = generate_dataset_AR(n, d, rho)
    datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)

    datasetAR_missing = generate_missingness(datasetAR, missingness_percentage)
    datasetBW_missing = generate_missingness(datasetBW, missingness_percentage)

    verify_dataset(datasetAR_missing)
    verify_dataset(datasetBW_missing)



    imputed_kNN_AR = kNN.impute(datasetAR_missing, neighbors, q)
    imputed_kNN_BW = kNN.impute(datasetBW_missing, neighbors, q)

    imputed_wNN_AR = wNN.impute(datasetAR_missing, neighbors, q, kernel_type, lambd)
    imputed_wNN_BW = wNN.impute(datasetBW_missing, neighbors, q, kernel_type, lambd)

    imputed_wNNC_AR = wNNC.impute(datasetAR, datasetAR_missing, neighbors, q, kernel_type, lambd)
    imputed_wNNC_BW = wNNC.impute(datasetBW, datasetBW_missing, neighbors, q, kernel_type, lambd)



    MSIE_AR_kNN += kNN.MSIE(datasetAR, imputed_kNN_AR)
    MAIE_AR_kNN += kNN.MAIE(datasetAR, imputed_kNN_AR)

    MSIE_BW_kNN += kNN.MSIE(datasetBW, imputed_kNN_BW)
    MAIE_BW_kNN += kNN.MAIE(datasetBW, imputed_kNN_BW)


    MSIE_AR_wNN += wNN.MSIE(datasetAR, imputed_wNN_AR)
    MAIE_AR_wNN += wNN.MAIE(datasetAR, imputed_wNN_AR)

    MSIE_BW_wNN += wNN.MSIE(datasetBW, imputed_wNN_BW)
    MAIE_BW_wNN += wNN.MAIE(datasetBW, imputed_wNN_BW)


    MSIE_AR_wNNC += wNNC.MSIE(datasetAR, imputed_wNNC_AR)
    MAIE_AR_wNNC += wNNC.MAIE(datasetAR, imputed_wNNC_AR)

    MSIE_BW_wNNC += wNNC.MSIE(datasetBW, imputed_wNNC_BW)
    MAIE_BW_wNNC += wNNC.MAIE(datasetBW, imputed_wNNC_BW)



print("******************************** AR ********************************")
print("                     kNN                 wNN                    wNNCorrelation")
print(f" MSIE    {MSIE_AR_kNN/samples}      {MSIE_AR_wNN/samples}       {MSIE_AR_wNNC/samples}")
print(f" MAIE    {MAIE_AR_kNN/samples}      {MAIE_AR_wNN/samples}       {MAIE_AR_wNNC/samples}")

print()
print()

print("******************************** BW ********************************")
print("                 kNN                     wNN                    wNNCorrelation")
print(f" MSIE    {MSIE_BW_kNN/samples}      {MSIE_BW_wNN/samples}       {MSIE_BW_wNNC/samples}")
print(f" MAIE    {MAIE_BW_kNN/samples}      {MAIE_BW_wNN/samples}       {MAIE_BW_wNNC/samples}")
