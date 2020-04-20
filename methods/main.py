import math
import copy
import numpy as np
from scipy.spatial import distance

from EED import EED
from EM import EM
from kNN import kNN, wNN, wNN_correlation
from ESD import ESD

from utils import is_pos_def, generate_dataset_AR, generate_dataset_blockwise, generate_missingness, verify_dataset, realDistances, generate_MAR, generate_MCAR, generate_NMAR

# *************************** EM *****************************
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

# model = EM()
#
# for _ in tqdm(range(samples)):
#     datasetAR = generate_dataset_AR(n, d, rho)
#     datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)
#
#     datasetAR, datasetAR_missing = generate_missingness(datasetAR, missingness_percentage)
#     datasetBW, datasetBW_missing = generate_missingness(datasetBW, missingness_percentage)
#
#     priors, mus, covs, imputed_dataset = model.impute(datasetAR_missing, 1)
#
#     print("MSIE AR => ", model.MSIE(datasetAR, imputed_dataset))
#     print("MAIE AR => ", model.MAIE(datasetAR, imputed_dataset))
#
#
#     priors, mus, covs, imputed_dataset = model.impute(datasetBW_missing, 1)
#
#     print("MSIE BW => ", model.MSIE(datasetBW, imputed_dataset))
#     print("MAIE BW => ", model.MAIE(datasetBW, imputed_dataset))


# ********************** kNN *********************************************

# samples = 1
#
# # GENERATE DATASETS
# n = 100
# d = 15
# #   AR
# rho = 0.9
# #   BW
# rho_w = 0.9
# rho_b = 0.1
# predictors_per_block = 5
#
# # MISSINGNESS
# missingness_percentage = 0.15
#
# # NEAREST NEIGHBOR
# neighbors = 10
# q = 2
# lambd = 1
# mC = 6
# kernels = ['Gaussian',
#            'Tricube']
# kernel_type = kernels[0]
# # ******************************************************************
#
# MSIE_AR_kNN = 0
# MAIE_AR_kNN = 0
#
# MSIE_BW_kNN = 0
# MAIE_BW_kNN = 0
#
#
# MSIE_AR_wNN = 0
# MAIE_AR_wNN = 0
#
# MSIE_BW_wNN = 0
# MAIE_BW_wNN = 0
#
#
# MSIE_AR_wNNC = 0
# MAIE_AR_wNNC = 0
#
# MSIE_BW_wNNC = 0
# MAIE_BW_wNNC = 0
#
#
# kNN = kNN()
# wNN = wNN()
# wNNC = wNN_correlation()
#
# for _ in tqdm(range(samples)):
#     datasetAR = generate_dataset_AR(n, d, rho)
#     datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)
#
#     datasetAR, datasetAR_missing = generate_missingness(datasetAR, missingness_percentage)
#     datasetBW, datasetBW_missing = generate_missingness(datasetBW, missingness_percentage)
#
#     verify_dataset(datasetAR_missing)
#     verify_dataset(datasetBW_missing)
#
#
#
#     imputed_kNN_AR = kNN.impute(datasetAR_missing, neighbors, q)
#     imputed_kNN_BW = kNN.impute(datasetBW_missing, neighbors, q)
#
#     imputed_wNN_AR = wNN.impute(datasetAR_missing, neighbors, q, kernel_type, lambd)
#     imputed_wNN_BW = wNN.impute(datasetBW_missing, neighbors, q, kernel_type, lambd)
#
#     imputed_wNNC_AR = wNNC.impute(datasetAR, datasetAR_missing, neighbors, q, kernel_type, lambd)
#     imputed_wNNC_BW = wNNC.impute(datasetBW, datasetBW_missing, neighbors, q, kernel_type, lambd)
#
#
#
#     MSIE_AR_kNN += kNN.MSIE(datasetAR, imputed_kNN_AR)
#     MAIE_AR_kNN += kNN.MAIE(datasetAR, imputed_kNN_AR)
#
#     MSIE_BW_kNN += kNN.MSIE(datasetBW, imputed_kNN_BW)
#     MAIE_BW_kNN += kNN.MAIE(datasetBW, imputed_kNN_BW)
#
#
#     MSIE_AR_wNN += wNN.MSIE(datasetAR, imputed_wNN_AR)
#     MAIE_AR_wNN += wNN.MAIE(datasetAR, imputed_wNN_AR)
#
#     MSIE_BW_wNN += wNN.MSIE(datasetBW, imputed_wNN_BW)
#     MAIE_BW_wNN += wNN.MAIE(datasetBW, imputed_wNN_BW)
#
#
#     MSIE_AR_wNNC += wNNC.MSIE(datasetAR, imputed_wNNC_AR)
#     MAIE_AR_wNNC += wNNC.MAIE(datasetAR, imputed_wNNC_AR)
#
#     MSIE_BW_wNNC += wNNC.MSIE(datasetBW, imputed_wNNC_BW)
#     MAIE_BW_wNNC += wNNC.MAIE(datasetBW, imputed_wNNC_BW)
#
#
#
# print("******************************** AR ********************************")
# print("                     kNN                 wNN                    wNNCorrelation")
# print(f" MSIE    {MSIE_AR_kNN/samples}      {MSIE_AR_wNN/samples}       {MSIE_AR_wNNC/samples}")
# print(f" MAIE    {MAIE_AR_kNN/samples}      {MAIE_AR_wNN/samples}       {MAIE_AR_wNNC/samples}")
#
# print()
# print()
#
# print("******************************** BW ********************************")
# print("                 kNN                     wNN                    wNNCorrelation")
# print(f" MSIE    {MSIE_BW_kNN/samples}      {MSIE_BW_wNN/samples}       {MSIE_BW_wNNC/samples}")
# print(f" MAIE    {MAIE_BW_kNN/samples}      {MAIE_BW_wNN/samples}       {MAIE_BW_wNNC/samples}")










# **************************** ESD *********************************
missingness_percentage = [0.1]
iterations = 1

# model = ESD()
#
# for mis in missingness_percentage:
#     print(f"ESD MISSINGNESS PERCENTAGE => {mis}")
#     RMSE = 0
#     for it in range(iterations):
#
#         dataset = generate_dataset_AR(n, d, rho)
#         dataset = (dataset - np.mean(dataset)) / np.std(dataset)
#         dataset, dataset_missing = generate_missingness(dataset, mis)
#
#         n, d = dataset_missing.shape
#
#         P = model.estimateDistances(copy.deepcopy(dataset_missing), 1)
#         R = realDistances(dataset)
#
#         s = 0
#         nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
#         for i in nan_indices:
#             for j in nan_indices:
#                 if i > j:
#                     s += (P[i][j] - R[i][j]) ** 2
#
#         count = len(nan_indices)
#         RMSE += math.sqrt(s/(count * n - count * (count + 1)/2))
#
#     print(f"RMSE => {RMSE/iterations}")
#     print()

# **************************** EED ************************


real_and_not_real = 0
real_and_imputed = 0
missingness_percentage = [0.25]
for mis in missingness_percentage:
    print("EED MISSINGNESS PERCENTAGE => ", mis)
    for it in range(1, 5):
        n = 100
        mean = np.array([-0.3, 0.1, 2])
        cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
        dataset = np.random.multivariate_normal(mean, cov, n)

        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_MAR(dataset, mis)

        n, d = dataset_missing.shape

        modelEM = EM()
        priors, mus, covs, imputed_dataset = modelEM.impute(dataset_missing, 1)

        # *****************************************************************************
        indices = np.array(np.where(np.any(np.isnan(dataset_missing), axis=1)))[0]

        real = []
        imputed = []
        not_real = []

        model = EED()

        for i, _ in enumerate(indices):
            for j, _ in enumerate(indices):
                if i != j:
                    not_real.append(model.estimateDistance(dataset_missing[indices[i]], dataset_missing[indices[j]], modelEM))
                    real.append(distance.euclidean(dataset[indices[i]], dataset[indices[j]]))
                    imputed.append(distance.euclidean(imputed_dataset[indices[i]], imputed_dataset[indices[j]]))

        not_real = np.array(not_real)
        real = np.array(real)
        imputed = np.array(imputed)


        from sklearn.metrics import mean_squared_error

        real_and_not_real += mean_squared_error(real, not_real)
        real_and_imputed += mean_squared_error(real, imputed)

print("NOT REAL AND REAL => ", real_and_not_real/it)
print("IMPUTED AND REAL => ", real_and_imputed/it)

print()
model = ESD()
RMSE = 0
for mis in missingness_percentage:
    print("ESD MISSINGNESS PERCENTAGE => ", mis)
    for it in range(1, 5):
        n = 100
        mean = np.array([-0.3, 0.1, 2])
        cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
        dataset = np.random.multivariate_normal(mean, cov, n)

        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_MAR(dataset, mis)

        n, d = dataset_missing.shape

        P = model.estimateDistances(copy.deepcopy(dataset_missing), 1)
        R = realDistances(dataset)

        s = 0
        nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
        for i in nan_indices:
            for j in nan_indices:
                if i > j:
                    s += (math.sqrt(P[i][j]) - math.sqrt(R[i][j])) ** 2

        count = len(nan_indices)
        RMSE += math.sqrt(s/(count * n - count * (count + 1)/2))
    print(f"RMSE => {RMSE/it}")
