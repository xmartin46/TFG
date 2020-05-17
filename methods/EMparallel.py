# Efficient EM Training of Gaussian Mixtures with Missing Data
# Authors: Olivier Delalleau, Aaron Courville, and Yosua Bengio

import time
import math
import copy
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import multiprocessing as mp
import matplotlib.pyplot as plt
from numpy.random import normal
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .utils import impute_methods


#Magic
from datetime import datetime as dt
import numpy as np
from functools import reduce

# ******************************** CLASSES ***********************************
class EMP(impute_methods):
    def __init__(self):
        self.n = None
        self.d = None
        self.n_gaussians = None
        self.priors = None
        self.mus = None
        self.covs = None

    def __C(self, dataset, probabilities, cov, cluster):
        C = np.zeros((self.d, self.d))

        for i in range(self.n):
            nan_indexes = np.isnan(dataset[i])
            if np.any(nan_indexes):
                # cov_mm = cov[nan_indexes, :][:, nan_indexes]
                # cov_mo = cov[nan_indexes, :][:, ~nan_indexes]
                # cov_oo = cov[~nan_indexes, :][:, ~nan_indexes]
                # cov_oo_inverse = np.linalg.pinv(cov_oo)
                #
                # aux = cov_mm - np.dot(cov_mo, np.dot(cov_oo_inverse, cov_mo.T))

                aux = np.linalg.inv(cov + 1e-6 * np.identity(cov.shape[0]))
                aux = np.linalg.inv(cov[nan_indexes, :][:, nan_indexes] + 1e-6 * np.identity(cov[nan_indexes, :][:, nan_indexes].shape[0]))

                C[nan_indexes, :][:, nan_indexes] += (probabilities[cluster][i] / (probabilities.sum(axis=1)[cluster] + 1e-308)) * aux

        return C

    def __worker1(self, dataset, priors, mus, covs, cluster, out_q):
        probs = np.zeros((self.n), dtype=float)
        for i in range(self.n):
            mu_cluster = mus[cluster]
            cov_cluster = covs[cluster]

            nan_indexes = np.isnan(dataset[i])
            mu_o = mu_cluster[~nan_indexes]
            cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]

            probs[i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo, allow_singular=True) # * priors[cluster]
            # print(i)
        out_q.put(probs)
        # print("Hi")

    def __e_step(self, dataset, priors, mus, covs, n_gaussians):
        probabilities = np.zeros((n_gaussians, self.n), dtype=float)

        out_q = mp.Queue()
        procs = []
        for cluster in range(n_gaussians):
            p = mp.Process(
                    target=self.__worker1,
                    args=(dataset, priors, mus, covs, cluster, out_q))
            procs.append(p)
            p.start()

        for cluster in range(n_gaussians):
            probabilities[cluster] = out_q.get()

        for p in procs:
            p.join()

        aux = probabilities.sum(axis=0)
        return probabilities/(aux + 1e-308)

    def __worker2(self, dataset, probabilities, priors, mus, covs, cluster, out_q):
        elem_belong_to_cluster = np.argmax(probabilities, axis=0)
        imputed_dataset = copy.deepcopy(dataset)

        # Expect missing values
        data_aux = copy.deepcopy(dataset)
        for i in range(self.n):
            if np.sum(np.isnan(dataset[i])) != 0:
                mu_cluster = mus[cluster]
                cov_cluster = covs[cluster]

                nan_indexes = np.isnan(dataset[i])
                mu_m = mu_cluster[nan_indexes]
                mu_o = mu_cluster[~nan_indexes]
                cov_mo = cov_cluster[nan_indexes, :][:, ~nan_indexes]
                cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                cov_oo_inverse = np.linalg.inv(cov_oo + 1e-6 * np.identity(cov_oo.shape[0]))

                aux = np.dot(cov_mo,
                                np.dot(cov_oo_inverse, (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
                nan_count = np.sum(nan_indexes)
                data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)

                if cluster == elem_belong_to_cluster[i]:
                    imputed_dataset[i] = data_aux[i]
            else:
                imputed_dataset[i] = dataset[i]

        new_mu = (data_aux * probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
        temp3 = (data_aux - new_mu) * probabilities.T[:,cluster][:,np.newaxis]
        temp4 = (data_aux - new_mu).T
        new_cov = np.dot(temp4, temp3)/(probabilities.sum(axis=1)[cluster] + 1e-308) + self.__C(dataset, probabilities, cov_cluster, cluster)

        out_q.put([new_mu, new_cov])

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):
        out_q = mp.Queue()
        procs = []
        for cluster in range(n_gaussians):
            p = mp.Process(
                    target=self.__worker2,
                    args=(dataset, probabilities, priors, mus, covs, cluster, out_q))
            procs.append(p)
            p.start()

        new_mus = []
        new_covs = []
        for cluster in range(n_gaussians):
            elems = out_q.get()
            new_mus.append(elems[0])
            new_covs.append(elems[1])

        for p in procs:
            p.join()

        priors = probabilities.sum(axis=1)/self.n

        return priors, np.array(new_mus), np.array(new_covs), 0

    def impute(self, dataset, n_gaussians, n_iters=10, epsilon=1e-20, init='kmeans', verbose=False):
        self.n, self.d = dataset.shape
        self.n_gaussians = n_gaussians

        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        mus = np.zeros((n_gaussians, self.d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)

        aux = np.nanmean(dataset, axis=0)
        for cluster in range(n_gaussians):
            mus[cluster] = aux

        for cluster in range(n_gaussians):
            # covs[cluster] = 1e-6 * np.identity(self.d)
            covs[cluster] = np.diag(np.nanvar(dataset, axis=0))


        for it in range(n_iters):
            mu_old = mus.copy()

            start = time.time()
            probabilities = self.__e_step(dataset, priors, mus, covs, n_gaussians)
            priors, mus, covs, imputed_dataset = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)
            print(time.time() - start)

            # convergence
            temp = 0
            for j in range(n_gaussians):
                temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
            temp = round(temp,20)
            if temp <= epsilon:
                break

        if verbose:
            print ("Iteration number = %d, stopping criterion = %.17f" %(it+1,temp))

        self.priors = priors
        self.mus = mus
        self.covs = covs





        elem_belong_to_cluster = np.argmax(probabilities, axis=0)
        imputed_dataset = copy.deepcopy(dataset)

        for cluster in range(n_gaussians):
            # Expect missing values
            data_aux = copy.deepcopy(dataset)
            for i in range(self.n):
                if np.sum(np.isnan(dataset[i])) != 0:
                    mu_cluster = mus[cluster]
                    cov_cluster = covs[cluster]

                    nan_indexes = np.isnan(dataset[i])
                    mu_m = mu_cluster[nan_indexes]
                    mu_o = mu_cluster[~nan_indexes]
                    cov_mo = cov_cluster[nan_indexes, :][:, ~nan_indexes]
                    cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                    cov_oo_inverse = np.linalg.inv(cov_oo + 1e-6 * np.identity(cov_oo.shape[0]))

                    aux = np.dot(cov_mo,
                                    np.dot(cov_oo_inverse, (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
                    nan_count = np.sum(nan_indexes)
                    data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)

                    if cluster == elem_belong_to_cluster[i]:
                        imputed_dataset[i] = data_aux[i]
                else:
                    imputed_dataset[i] = dataset[i]











        return priors, mus, covs, imputed_dataset

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

# class EM(impute_methods):
#     """
#     this algorithm just require to lean the Gauss distribution elements 'mu'
#     and 'sigma'
#     """
#     def __init__(self, max_iter = 100, theta = 1e-5, normalizer = 'min_max'):
#         self.max_iter = max_iter
#         self.theta = theta
#
#     def _init_parameters(self, X):
#         rows, cols = X.shape
#         mu_init = np.nanmean(X, axis = 0)
#         sigma_init = np.zeros((cols, cols))
#         for i in range(cols):
#             for j in range(i, cols):
#                 vec_col = X[: , [i, j]]
#                 vec_col = vec_col[~np.any(np.isnan(vec_col), axis = 1),: ].T
#                 if len(vec_col) > 0:
#                     cov = np.cov(vec_col)
#                     cov = cov[0, 1]
#                     sigma_init[i, j] = cov
#                     sigma_init[j, i] = cov
#
#                 else :
#                     sigma_init[i, j] = 1.0
#                     sigma_init[j, i] = 1.0
#
#         return mu_init, sigma_init
#
#     def _e_step(self, mu, sigma, X):
#         samples, _ = X.shape
#         for sample in range(samples):
#             if np.any(np.isnan(X[sample,: ])):
#                 loc_nan = np.isnan(X[sample,: ])
#                 new_mu = np.dot(sigma[loc_nan,: ][: , ~loc_nan], np.dot(np.linalg.pinv(sigma[~loc_nan,: ][: , ~loc_nan]), (X[sample, ~loc_nan] - mu[~loc_nan])[: , np.newaxis]))
#                 nan_count = np.sum(loc_nan)
#                 X[sample, loc_nan] = mu[loc_nan] + new_mu.reshape(1, nan_count)
#
#         return X
#
#     def _m_step(self, X):
#         rows, cols = X.shape
#         mu = np.mean(X, axis = 0)
#         sigma = np.cov(X.T)
#         tmp_theta = -0.5 * rows * (cols * np.log(2 * np.pi) + np.log(np.linalg.det(sigma)))
#
#         return mu, sigma, tmp_theta
#
#     def solve(self, X):
#         mu, sigma = self._init_parameters(X)
#         complete_X, updated_X = None, None
#         rows, _ = X.shape
#         theta = -np.inf
#         for iter in range(self.max_iter):
#             updated_X = self._e_step(mu = mu, sigma = sigma, X = copy.copy(X))
#             mu, sigma, tmp_theta = self._m_step(updated_X)
#             for i in range(rows):
#                 tmp_theta -= 0.5 * np.dot((updated_X[i,: ] - mu), np.dot(np.linalg.pinv(sigma), (updated_X[i,: ] - mu)[: , np.newaxis]))
#
#             if abs(tmp_theta - theta) < self.theta:
#                 complete_X = updated_X
#                 break;
#             else:
#                 theta = tmp_theta
#         else:
#             complete_X = updated_X
#
#         return complete_X
