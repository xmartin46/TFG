# Euclidean distance estimation in incompelte datasets
# Authors: Diego P.P. Mesquita, João P.P. Gomes, Amauri H. SOuza Junior, Juvêncio S. Nobre

import time
import math
import copy
import random
import scipy.special
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import normal
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .ESD import ESD
from .EM import EM

# ******************************** CLASSES ***********************************
class EED:
    def __init__(self):
        pass

    def __VarZ(self, Xi, Xj, GMM):
        # Initialization
        Mi = np.isnan(Xi)
        Oi = ~np.isnan(Xi)

        Mj = np.isnan(Xj)
        Oj = ~np.isnan(Xj)

        eta_estimation = 0

        # Condition each of the GMM components on the observed values of both Xi and Xj
        mus_i = np.zeros((GMM.n_gaussians, np.sum(Mi)), dtype=float)
        covs_i = np.zeros((GMM.n_gaussians, np.sum(Mi), np.sum(Mi)), dtype=float)

        mus_j = np.zeros((GMM.n_gaussians, np.sum(Mj)), dtype=float)
        covs_j = np.zeros((GMM.n_gaussians, np.sum(Mj), np.sum(Mj)), dtype=float)

        for cluster in range(GMM.n_gaussians):
            mu_cluster = GMM.mus[cluster]
            cov_cluster = GMM.covs[cluster]

            # mu_i
            mu_m = mu_cluster[Mi]
            mu_o = mu_cluster[Oi]
            cov_mo = cov_cluster[Mi, :][:, Oi]
            cov_oo = cov_cluster[Oi, :][:, Oi]
            cov_oo_inverse = np.linalg.pinv(cov_oo)

            aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (Xi[Oi] - mu_o)[:,np.newaxis]))
            nan_count = np.sum(Mi)
            mus_i[cluster] = mu_m + aux.reshape(1, nan_count)

            # cov_i
            aux = np.linalg.pinv(cov_cluster)
            covs_i[cluster] = np.linalg.pinv(aux[Mi, :][:, Mi])


            # mu_j
            mu_m = mu_cluster[Mj]
            mu_o = mu_cluster[Oj]
            cov_mo = cov_cluster[Mj, :][:, Oj]
            cov_oo = cov_cluster[Oj, :][:, Oj]
            cov_oo_inverse = np.linalg.pinv(cov_oo)

            aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (Xj[Oj] - mu_o)[:,np.newaxis]))
            nan_count = np.sum(Mj)
            mus_j[cluster] = mu_m + aux.reshape(1, nan_count)

            # cov_j
            aux = np.linalg.pinv(cov_cluster)
            covs_j[cluster] = np.linalg.pinv(aux[Mj, :][:, Mj])

        # Compute padded conditional mean vectors and conditional covariance
        # matrices of Xi - Xj for ecah GMM component
        mu = np.zeros((GMM.n_gaussians, GMM.d), dtype=float)
        co = np.zeros((GMM.n_gaussians, GMM.d, GMM.d), dtype=float)

        for cluster in range(GMM.n_gaussians):
            mu[cluster, Oi] += Xi[Oi]
            mu[cluster, Mi] += mus_i[cluster]

            mu[cluster, Oj] -= Xj[Oj]
            mu[cluster, Mj] -= mus_j[cluster]

            ri = 0
            for row, b1 in enumerate(Mi):
                if b1:
                    ci = 0
                    for col, b2 in enumerate(Mi):
                        if b2:
                            co[cluster][row][col] += covs_i[cluster][ri][ci]
                            ci += 1
                    ri += 1

            rj = 0
            for row, b1 in enumerate(Mj):
                if b1:
                    cj = 0
                    for col, b2 in enumerate(Mj):
                        if b2:
                            co[cluster][row][col] += covs_j[cluster][rj][cj]
                            cj += 1
                    rj += 1

        for d in range(GMM.d):
            m = 0
            s = 0
            for cluster in range(GMM.n_gaussians):
                m += GMM.priors[cluster] * mu[cluster][d]
                s += GMM.priors[cluster] * ((mu[cluster][d] ** 2) + co[cluster][d][d])
            v = s - (m ** 2)
            eta_estimation += 4 * (m ** 2) * v + 2 * (v ** 2)

        return eta_estimation

    def estimateDistances(self, dataset_missing):
        modelEM = EM()
        priors, mus, covs, imputed_dataset = modelEM.impute(dataset_missing, 1, n_iters=30, epsilon=1e-4)

        modelESD = ESD()
        EzALL = modelESD.estimateDistances(dataset_missing, 1)

        P = np.zeros((modelEM.n, modelEM.n), dtype=float)

        for i in range(modelEM.n):
            for j in range(modelEM.n):
                if i > j:
                    if np.any(np.isnan(dataset_missing[i])) or np.any(np.isnan(dataset_missing[j])):
                        Varz = self.__VarZ(dataset_missing[i], dataset_missing[j], modelEM)
                        Ez = EzALL[i][j]
                        m = (Ez ** 2)/Varz
                        P[i][j] = np.exp(scipy.special.gammaln(m + 0.5) - scipy.special.gammaln(m)) * math.sqrt(Ez/m)
                    else:
                        P[i][j] = distance.euclidean(dataset_missing[i], dataset_missing[j])

        return P


# class EED:
#     def __init__(self):
#         pass
#
#     def estimateDistance(self, Xi, Xj, GMM):
#         # Initialization
#         nan_indexes_i = np.isnan(Xi)
#         nan_indexes_j = np.isnan(Xj)
#
#         eta_estimation = 0
#
#         # Condition each of the GMM components on the observed values of both Xi and Xj
#         mus_i = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_i)), dtype=float)
#         covs_i = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_i), np.sum(nan_indexes_i)), dtype=float)
#
#         mus_j = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_j)), dtype=float)
#         covs_j = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_j), np.sum(nan_indexes_j)), dtype=float)
#
#         for cluster in range(GMM.n_gaussians):
#             cov_c = GMM.covs[cluster]
#             mu_c = GMM.mus[cluster]
#
#             # mu_i
#             cov_i_oo = cov_c[~nan_indexes_i, :][:, ~nan_indexes_i]
#             cov_i_mo = cov_c[nan_indexes_i, :][:, ~nan_indexes_i]
#
#             aux = np.dot(cov_i_mo, np.dot(np.linalg.inv(cov_i_oo), (Xi[~nan_indexes_i] - mu_c[~nan_indexes_i])[:,np.newaxis]))
#             nan_count = np.sum(nan_indexes_i)
#             mus_i[cluster] = mu_c[nan_indexes_i] + aux.reshape(1, nan_count)
#
#
#             # cov_i
#             aux = np.linalg.inv(cov_c)
#             covs_i[cluster] = np.linalg.inv(aux[nan_indexes_i, :][:, nan_indexes_i])
#
#
#             # mu_j
#             cov_j_oo = cov_c[~nan_indexes_j, :][:, ~nan_indexes_j]
#             cov_j_mo = cov_c[nan_indexes_j, :][:, ~nan_indexes_j]
#
#             aux = np.dot(cov_j_mo, np.dot(np.linalg.inv(cov_j_oo), (Xj[~nan_indexes_j] - mu_c[~nan_indexes_j])[:,np.newaxis]))
#             nan_count = np.sum(nan_indexes_j)
#             mus_j[cluster] = mu_c[nan_indexes_j] + aux.reshape(1, nan_count)
#
#
#             # cov_j
#             aux = np.linalg.inv(cov_c)
#             covs_j[cluster] = np.linalg.inv(aux[nan_indexes_j, :][:, nan_indexes_j])
#
#         # Compute padded conditional mean vectors and conditional covariance matrices of Xi - Xj for each GMM component
#         new_mus = np.zeros((GMM.n_gaussians, GMM.d), dtype=float)
#         new_covs = np.zeros((GMM.n_gaussians, GMM.d, GMM.d), dtype=float)
#
#         for cluster in range(GMM.n_gaussians):
#             new_mus[cluster][~nan_indexes_i] += Xi[~nan_indexes_i]
#             new_mus[cluster][nan_indexes_i] += mus_i[cluster]
#
#             new_mus[cluster][~nan_indexes_j] -= Xj[~nan_indexes_j]
#             new_mus[cluster][nan_indexes_j] -= mus_j[cluster]
#
#             ind1 = 0
#             for i1, b1 in enumerate(nan_indexes_i):
#                 if b1:
#                     ind2 = 0
#                     for i2, b2 in enumerate(nan_indexes_i):
#                         if b2:
#                             new_covs[cluster][i1][i2] += covs_i[cluster][ind1][ind2]
#                             ind2 += 1
#                     ind1 += 1
#
#             ind1 = 0
#             for i1, b1 in enumerate(nan_indexes_j):
#                 if b1:
#                     ind2 = 0
#                     for i2, b2 in enumerate(nan_indexes_j):
#                         if b2:
#                             new_covs[cluster][i1][i2] += covs_j[cluster][ind1][ind2]
#                             ind2 += 1
#                     ind1 += 1
#
#         # Compute eta estimation
#         for d in range(GMM.d):
#             m = 0
#             s = 0
#             for cluster in range(GMM.n_gaussians):
#                 prior_c = GMM.priors[cluster]
#                 m += prior_c * new_mus[cluster][d]
#                 s += prior_c * ((new_mus[cluster][d] ** 2) + new_covs[cluster][d][d])
#             v = s - (m ** 2)
#             eta_estimation += 4 * (m ** 2) * v + 2 * (v ** 2)
#
#         return eta_estimation
