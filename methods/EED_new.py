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
from numpy.random import normal
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

import scipy.special
from ESD import ESD

# ******************************** CLASSES ***********************************
class EED:
    def __init__(self):
        pass

    def estimateDistance(self, Xi, Xj, GMM):
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

            mu[cluster, Oj] += Xj[Oj]
            mu[cluster, Mj] += mus_j[cluster]

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

from EM import EM
from utils import generate_MAR, generate_MCAR
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from sklearn import datasets


real_and_not_real = 0
real_and_imputed = 0
missingness_percentage = [0.5]
iterations = 10
for mis in missingness_percentage:
    print("EED MISSINGNESS PERCENTAGE => ", mis)
    for it in range(iterations):
        # n = 100
        # mean = np.array([-0.3, 0.1, 2])
        # cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
        # dataset = np.random.multivariate_normal(mean, cov, n)

        from auto_mpg import mpg
        dataset = mpg

        # iris = datasets.load_iris()
        # dataset = iris.data

        # from housing import housing
        # dataset = housing

        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_mine(dataset, mis)

        n, d = dataset_missing.shape
        print(dataset_missing.shape)

        modelEM = EM()
        k = BIC(dataset)
        print("N_COMPONENTS => ", k)
        priors, mus, covs, imputed_dataset = modelEM.impute(dataset_missing, k)

        # *****************************************************************************
        indices = np.array(np.where(np.any(np.isnan(dataset_missing), axis=1)))[0]

        real = []
        not_real = []
        np.seterr('raise')
        model = EED()
        modelESD = ESD()
        EzALL = modelESD.estimateDistances(copy.deepcopy(dataset_missing), k)
        for i in range(n):
            for j in range(n):
                if i > j:
                    if np.any(np.isnan(dataset_missing[i])) or np.any(np.isnan(dataset_missing[j])):
                        Varz = model.estimateDistance(dataset_missing[i], dataset_missing[j], modelEM)
                        modelESD = ESD()
                        Ez = EzALL[i][j]
                        m = (Ez ** 2)/Varz
                        f = np.exp(scipy.special.gammaln(m + 0.5) - scipy.special.gammaln(m))
                        dist = f * math.sqrt(Ez/m)

                        not_real.append(dist)
                        real.append(distance.euclidean(dataset[i], dataset[j]))
                    else:
                        not_real.append(distance.euclidean(dataset[i], dataset[j]))
                        real.append(distance.euclidean(dataset[i], dataset[j]))


        not_real = np.array(not_real)
        real = np.array(real)

        real_and_not_real += math.sqrt(mean_squared_error(real, not_real))
        print(math.sqrt(mean_squared_error(real, not_real)))

print("NOT REAL AND REAL => ", real_and_not_real/iterations)
