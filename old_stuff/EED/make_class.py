# Euclidean distance estimation in incompelte datasets
# Authors: Diego P.P. Mesquita, João P.P. Gomes, Amauri H. SOuza Junior, Juvêncio S. Nobre

import time
import math
import copy
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import normal
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# ******************************** CLASSES ***********************************
class impute_methods:
    def __init__(self):
        pass

    def MSIE(self, real, imputed):
        return np.sum(abs(real - imputed) ** 2)/np.count_nonzero(real - imputed)

    def MAIE(self, real, imputed):
        return np.sum(abs(real - imputed))/np.count_nonzero(real - imputed)

def EED(Xi, Xj, GMM):
    # Initialization
    nan_indexes_i = np.isnan(Xi)
    nan_indexes_j = np.isnan(Xj)

    eta_estimation = 0

    # Condition each of the GMM components on the observed values of both Xi and Xj
    mus_i = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_i)), dtype=float)
    covs_i = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_i), np.sum(nan_indexes_i)), dtype=float)

    mus_j = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_j)), dtype=float)
    covs_j = np.zeros((GMM.n_gaussians, np.sum(nan_indexes_j), np.sum(nan_indexes_j)), dtype=float)

    for cluster in range(GMM.n_gaussians):
        cov_c = GMM.covs[cluster]
        mu_c = GMM.mus[cluster]

        # mu_i
        cov_i_oo = cov_c[~nan_indexes_i, :][:, ~nan_indexes_i]
        cov_i_mo = cov_c[nan_indexes_i, :][:, ~nan_indexes_i]

        aux = np.dot(cov_i_mo, np.dot(np.linalg.inv(cov_i_oo), (Xi[~nan_indexes_i] - mu_c[~nan_indexes_i])[:,np.newaxis]))
        nan_count = np.sum(nan_indexes_i)
        mus_i[cluster] = mu_c[nan_indexes_i] + aux.reshape(1, nan_count)

        # cov_i
        aux = np.linalg.inv(cov_c)
        covs_i[cluster] = np.linalg.inv(aux[nan_indexes_i, :][:, nan_indexes_i])

        # mu_j
        cov_j_oo = cov_c[~nan_indexes_j, :][:, ~nan_indexes_j]
        cov_j_mo = cov_c[nan_indexes_j, :][:, ~nan_indexes_j]

        aux = np.dot(cov_j_mo, np.dot(np.linalg.inv(cov_j_oo), (Xj[~nan_indexes_j] - mu_c[~nan_indexes_j])[:,np.newaxis]))
        nan_count = np.sum(nan_indexes_j)
        mus_j[cluster] = mu_c[nan_indexes_j] + aux.reshape(1, nan_count)

        # cov_j
        aux = np.linalg.inv(cov_c)
        covs_j[cluster] = np.linalg.inv(aux[nan_indexes_j, :][:, nan_indexes_j])

    # Compute padded conditional mean vectors and conditional covariance matrices of Xi - Xj for each GMM component
    new_mus = np.zeros((GMM.n_gaussians, GMM.d), dtype=float)
    new_covs = np.zeros((GMM.n_gaussians, GMM.d, GMM.d), dtype=float)

    for cluster in range(GMM.n_gaussians):
        new_mus[cluster][~nan_indexes_i] += Xi[~nan_indexes_i]
        new_mus[cluster][nan_indexes_i] += mus_i[cluster]

        new_mus[cluster][~nan_indexes_j] -= Xj[~nan_indexes_j]
        new_mus[cluster][nan_indexes_j] -= mus_j[cluster]

        ind1 = 0
        for i1, b1 in enumerate(nan_indexes_i):
            if b1:
                ind2 = 0
                for i2, b2 in enumerate(nan_indexes_i):
                    if b2:
                        new_covs[cluster][i1][i2] += covs_i[cluster][ind1][ind2]
                        ind2 += 1
                ind1 += 1

        ind1 = 0
        for i1, b1 in enumerate(nan_indexes_j):
            if b1:
                ind2 = 0
                for i2, b2 in enumerate(nan_indexes_j):
                    if b2:
                        new_covs[cluster][i1][i2] += covs_j[cluster][ind1][ind2]
                        ind2 += 1
                ind1 += 1

    # Compute eta estimation
    for d in range(GMM.d):
        m = 0
        s = 0
        for cluster in range(GMM.n_gaussians):
            prior_c = GMM.priors[cluster]
            m += prior_c * new_mus[cluster][d]
            s += prior_c * ((new_mus[cluster][d] ** 2) + new_covs[cluster][d][d])
        v = s - (m ** 2)
        eta_estimation += 4 * (m ** 2) * v + 2 * (v ** 2)

    return eta_estimation
