import math
import copy
import numpy as np
from scipy.spatial import distance

from EED import EED
from EM import EM
from kNN import kNN, wNN, wNN_correlation
from ESD import ESD

from utils import is_pos_def, generate_dataset_AR, generate_dataset_blockwise, generate_missingness, verify_dataset, realDistances, generate_MAR, generate_MCAR, generate_NMAR

from sklearn import datasets

from tqdm import tqdm

missingness_percentage = [0.1, 0.3, 0.5]
iterations = 10

model = ESD()

import random

def generate_mine(dataset, mis):
    dataset_copy = copy.deepcopy(dataset)
    indexes = random.sample(range(dataset.shape[0]), math.floor(mis * dataset.shape[0]))
    # print(indexes)
    for i in indexes:
        num = random.sample(range(math.ceil(dataset.shape[1]/3) + 1), 1)[0]
        # print(num)
        for j in random.sample(range(dataset.shape[1]), num):
            dataset[i][j] = np.nan

    return dataset_copy, dataset

def generate_p(dataset, mis):
    n, d = dataset.shape
    dataset_real = copy.deepcopy(dataset)

    for i in range(n):
        for j in range(d):
            if random.random() < mis:
                dataset[i][j] = float("nan")
    dataset = dataset.flatten()
    dataset = np.array(dataset)
    dataset = np.split(dataset, n)

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(dataset)), axis=1)))[0]

    indices.sort()
    indices = np.flip(indices)

    for i in indices:
        dataset.pop(i)
        dataset_real = np.delete(dataset_real, i, axis=0)

    return dataset_real, np.array(dataset)

for mis in missingness_percentage:
    print(f"ESD MISSINGNESS PERCENTAGE => {mis}")
    RMSE = 0
    real_and_not_real = 0
    for it in range(iterations):

        # n = 100
        # mean = np.array([-0.3, 0.1, 2])
        # cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
        # dataset = np.random.multivariate_normal(mean, cov, n)

        iris = datasets.load_iris()
        dataset = iris.data

        # from ecoli import ecoli
        # dataset = ecoli

        # from wine import wine
        # dataset = wine

        # from auto_mpg import mpg
        # dataset = mpg

        # from haberman import hab
        # dataset = hab
        #
        # from parkinsons import parkinson
        # dataset = parkinson

        # from glass import glass
        # dataset = glass

        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_mine(dataset, mis)


        n, d = dataset_missing.shape

        P = model.estimateDistances(copy.deepcopy(dataset_missing), 1)
        R = realDistances(dataset)

        s = 0
        nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
        for i in range(n):
            for j in range(n):
                if i > j:
                    s += (math.sqrt(P[i][j]) - math.sqrt(R[i][j])) ** 2

        count = len(nan_indices)
        RMSE += math.sqrt(s/(count * n - count * (count + 1)/2))
        # print(math.sqrt(s/(count * n - count * (count + 1)/2)))

        from sklearn.metrics import mean_squared_error
        real = []
        not_real = []
        for i in range(n):
            for j in range(n):
                if i > j:
                    not_real.append(math.sqrt(P[i][j]))
                    real.append(math.sqrt(R[i][j]))


        not_real = np.array(not_real)
        real = np.array(real)

        real_and_not_real += math.sqrt(mean_squared_error(real, not_real))
        print(math.sqrt(mean_squared_error(real, not_real)))

    print(f"REAL RMSE => {real_and_not_real/iterations}")
    print(f"RMSE => {RMSE/iterations}")
    print()











#
#
#
#
# import time
# import math
# import copy
# import random
# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from numpy.random import normal
# from sklearn.cluster import KMeans
# from scipy.stats import multivariate_normal
# from sklearn.mixture import GaussianMixture
#
# from utils import impute_methods
#
# from tqdm import tqdm
#
# class EM(impute_methods):
#     def __init__(self):
#         self.n = None
#         self.d = None
#         self.n_gaussians = None
#         self.priors = None
#         self.mus = None
#         self.covs = None
#
#     def __e_step(self, dataset, priors, mus, covs, n_gaussians):
#         probabilities = np.zeros((n_gaussians, self.n), dtype=float)
#
#         for cluster in range(n_gaussians):
#             for i in range(self.n):
#                 mu_cluster = mus[cluster]
#                 cov_cluster = covs[cluster]
#
#                 nan_indexes = np.isnan(dataset[i])
#                 mu_o = mu_cluster[~nan_indexes]
#                 cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
#
#                 probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo, allow_singular=True)
#
#         aux = probabilities.sum(axis=0)
#
#         return probabilities/(aux + 1e-308)
#
#     def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):
#         elem_belong_to_cluster = np.argmax(probabilities, axis=0)
#
#         imputed_dataset = copy.deepcopy(dataset)
#
#         for cluster in range(n_gaussians):
#             # Expect missing values
#             data_aux = copy.deepcopy(dataset)
#             for i in range(self.n):
#                 mu_cluster = mus[cluster]
#                 cov_cluster = covs[cluster]
#
#                 nan_indexes = np.isnan(dataset[i])
#                 mu_m = mu_cluster[nan_indexes]
#                 mu_o = mu_cluster[~nan_indexes]
#                 cov_mo = cov_cluster[nan_indexes, :][:, ~nan_indexes]
#                 cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
#                 cov_oo_inverse = np.linalg.pinv(cov_oo)
#
#                 aux = np.dot(cov_mo,
#                                 np.dot(cov_oo_inverse, (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
#                 nan_count = np.sum(nan_indexes)
#                 data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)
#
#                 if cluster == elem_belong_to_cluster[i]:
#                     imputed_dataset[i] = data_aux[i]
#
#             mus[cluster] = (data_aux *  probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
#             temp3 = (data_aux - mus[cluster])*  probabilities.T[:,cluster][:,np.newaxis]
#             temp4 = (data_aux - mus[cluster]).T
#             covs[cluster] = (np.dot(temp4, temp3))/(probabilities.sum(axis=1)[cluster] + 1e-308)
#
#         priors = probabilities.sum(axis=1)/self.n
#
#         return priors, mus, covs, imputed_dataset
#
#     def impute(self, dataset, n_gaussians, n_iters=300, epsilon=1e-5, init='kmeans', verbose=True):
#         self.n, self.d = dataset.shape
#         self.n_gaussians = n_gaussians
#
#         priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
#         mus = np.zeros((n_gaussians, self.d), dtype=float)
#         covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)
#
#
#
#
#         indices = np.array(np.where(np.all(~np.isnan(np.array(dataset)), axis=1)))[0]
#         if init == 'kmeans' and len(indices) > 0:
#             data_for_kmeans = dataset[indices]
#             kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)
#             mus = kmeans.cluster_centers_
#         elif len(indices) > 0:
#             for cluster in range(n_gaussians):
#                 mus[cluster] = dataset[indices[cluster]]
#         else:
#             for cluster in range(n_gaussians):
#                 mus[cluster] = np.random.rand(self.d) * 100
#
#         for cluster in range(n_gaussians):
#             covs[cluster] = np.identity(self.d)
#
#
#
#
#         for it in tqdm(range(n_iters)):
#             mu_old = mus.copy()
#
#             probabilities = self.__e_step(dataset, priors, mus, covs, n_gaussians)
#             priors, mus, covs, imputed_dataset = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)
#
#             # convergence
#             temp = 0
#             for j in range(n_gaussians):
#                 temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
#             temp = round(temp,20)
#             if temp <= epsilon:
#                 break
#
#         if verbose:
#             print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))
#
#         self.priors = priors
#         self.mus = mus
#         self.covs = covs
#
#         return priors, mus, covs, imputed_dataset
#
#     def MSIE(self, real, dataset):
#         return super().MSIE(real, dataset)
#
#     def MAIE(self, real, dataset):
#         return super().MAIE(real, dataset)
#
# # **************************** EED ************************
#
# from auto_mpg import mpg
#
# real_and_not_real = 0
# real_and_imputed = 0
# # missingness_percentage = [0.01]
#
# for mis in missingness_percentage:
#     print("EED MISSINGNESS PERCENTAGE => ", mis)
#
#     for it in range(iterations):
#         # n = 100
#         # mean = np.array([-0.3, 0.1, 2])
#         # cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
#         # dataset = np.random.multivariate_normal(mean, cov, n)
#
#         iris = datasets.load_iris()
#         dataset = iris.data
#
#         # dataset = mpg
#
#         dataset = (dataset - np.mean(dataset)) / np.std(dataset)
#         dataset, dataset_missing = generate_mine(dataset, mis)
#
#         n, d = dataset_missing.shape
#         print(n)
#         modelEM = EM()
#         priors, mus, covs, imputed_dataset = modelEM.impute(dataset_missing, 1)
#
#         # *****************************************************************************
#         indices = np.array(np.where(np.any(np.isnan(dataset_missing), axis=1)))[0]
#
#         real = []
#         imputed = []
#         not_real = []
#
#         model = EED()
#
#         for i in tqdm(range(n)):
#             for j in range(n):
#                 if i > j:
#                     not_real.append(model.estimateDistance(dataset_missing[i], dataset_missing[j], modelEM))
#                     real.append(distance.euclidean(dataset[i], dataset[j]))
#                     # imputed.append(distance.euclidean(imputed_dataset[i], imputed_dataset[j]))
#
#         not_real = np.array(not_real)
#         real = np.array(real)
#         # imputed = np.array(imputed)
#
#         from sklearn.metrics import mean_squared_error
#
#         real_and_not_real += mean_squared_error(real, not_real)
#         # real_and_imputed += mean_squared_error(real, imputed)
#
# print("NOT REAL AND REAL => ", real_and_not_real/iterations)
# # print("IMPUTED AND REAL => ", real_and_imputed/it)
