# Efficient EM Training of Gaussian Mixtures with Missing Data
# Authors: Olivier Delalleau, Aaron Courville, and Yosua Bengio

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

from .utils import impute_methods


#Magic
from datetime import datetime as dt
import numpy as np
from functools import reduce

# ******************************** CLASSES ***********************************
class EM_estimate(impute_methods):
    def __init__(self):
        self.n = None
        self.d = None

    def __e_step(self, dataset, priors, mus, covs, n_gaussians):
        probabilities = np.zeros((n_gaussians, self.n), dtype=float)

        for cluster in range(n_gaussians):
            probabilities[cluster] = multivariate_normal.pdf(dataset, mean=mus[cluster], cov=covs[cluster], allow_singular=True)
        aux = probabilities.sum(axis=0)

        return probabilities/(aux + 1e-308)

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):

        for cluster in range(n_gaussians):
            # do = np.zeros((self.d, self.d), dtype=float)
            # mus[cluster] = np.dot(probabilities[cluster], dataset)/sum(probabilities[cluster])
            #
            # for i in range(self.n):
            #     z = np.array(dataset[i] - mus[cluster])
            #     z = z.reshape(len(z), 1)
            #     a = z * probabilities[cluster][i]
            #     a = a.reshape(len(a), 1)
            #     do += np.dot(a, z.T)
            #
            # covs[cluster] = do/probabilities.sum(axis=1)[cluster]

            temp3 = (dataset -mus[cluster]) * probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (dataset -mus[cluster]).T
            covs[cluster] = np.dot(temp4,temp3)  / probabilities.sum(axis=1)[cluster]
            mus[cluster] = (dataset * probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/ probabilities.sum(axis=1)[cluster]

        priors = probabilities.sum(axis=1)/self.n
        return priors, mus, covs

    def estimate(self, dataset, n_gaussians, n_iters=200, epsilon=1e-20):
        self.n, self.d = dataset.shape

        # K-Means initialization
        kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(dataset)
        mus = kmeans.cluster_centers_

        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        # mus = np.zeros((n_gaussians, d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)

        for cluster in range(n_gaussians):
            # mus[cluster] = dataset[cluster]
            covs[cluster] = np.identity(self.d)

        for it in range(n_iters):
            mu_old = mus.copy()
            # probabilities size: n_gaussians x n
            probabilities = self.__e_step(dataset, priors, mus, covs, n_gaussians)

            priors, mus, covs = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

            #convergence?
            temp = 0
            for j in range(n_gaussians):
                temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
            temp = round(temp,20)
            if temp <= epsilon:
                print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))
                break

        return priors, mus, covs

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

# class EM(impute_methods):
#     def __init__(self):
#         self.n = None
#         self.d = None
#         self.n_gaussians = None
#         self.priors = None
#         self.mus = None
#         self.covs = None
#
#     def __C(self, dataset, probabilities, cov, cluster):
#         C = np.zeros((self.d, self.d))
#
#         for i in range(self.n):
#             nan_indexes = np.isnan(dataset[i])
#             if np.any(nan_indexes):
#                 # cov_mm = cov[nan_indexes, :][:, nan_indexes]
#                 # cov_mo = cov[nan_indexes, :][:, ~nan_indexes]
#                 # cov_oo = cov[~nan_indexes, :][:, ~nan_indexes]
#                 # cov_oo_inverse = np.linalg.pinv(cov_oo)
#                 #
#                 # aux = cov_mm - np.dot(cov_mo, np.dot(cov_oo_inverse, cov_mo.T))
#
#                 aux = np.linalg.pinv(cov)
#                 aux = np.linalg.pinv(cov[nan_indexes, :][:, nan_indexes])
#
#                 C[nan_indexes, :][:, nan_indexes] += (probabilities[cluster][i] / (probabilities.sum(axis=1)[cluster] + 1e-308)) * aux
#
#         return C
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
#             covs[cluster] = (np.dot(temp4, temp3) + 0 * self.__C(dataset, probabilities, cov_cluster, cluster))/(probabilities.sum(axis=1)[cluster] + 1e-308)

        priors = probabilities.sum(axis=1)/self.n

        return priors, mus, covs, imputed_dataset

    def impute(self, dataset, n_gaussians, n_iters=100, epsilon=1e-20, init='kmeans', verbose=False):
        self.n, self.d = dataset.shape
        self.n_gaussians = n_gaussians

        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        mus = np.zeros((n_gaussians, self.d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)




        indices = np.array(np.where(np.all(~np.isnan(np.array(dataset)), axis=1)))[0]
        if init == 'kmeans' and len(indices) > 0:
            data_for_kmeans = dataset[indices]
            kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)
            mus = kmeans.cluster_centers_
        elif len(indices) > 0:
            for cluster in range(n_gaussians):
                mus[cluster] = dataset[indices[cluster]]
        else:
            for cluster in range(n_gaussians):
                mus[cluster] = np.random.rand(self.d) * 100

        for cluster in range(n_gaussians):
            covs[cluster] = np.identity(self.d)




        for it in range(n_iters):
            mu_old = mus.copy()

            probabilities = self.__e_step(dataset, priors, mus, covs, n_gaussians)
            priors, mus, covs, imputed_dataset = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

            # convergence
            temp = 0
            for j in range(n_gaussians):
                temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
            temp = round(temp,20)
            if temp <= epsilon:
                break

        if verbose:
            print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))

        self.priors = priors
        self.mus = mus
        self.covs = covs

        return priors, mus, covs, imputed_dataset

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)


class EM(impute_methods):
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

    def __e_step(self, dataset, priors, mus, covs, n_gaussians):
        probabilities = np.zeros((n_gaussians, self.n), dtype=float)

        for cluster in range(n_gaussians):
            for i in range(self.n):
                mu_cluster = mus[cluster]
                cov_cluster = covs[cluster]

                nan_indexes = np.isnan(dataset[i])
                mu_o = mu_cluster[~nan_indexes]
                cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]

                probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo, allow_singular=True) # * priors[cluster]

        aux = probabilities.sum(axis=0)

        return probabilities/(aux + 1e-308)

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):
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

            mus[cluster] = (data_aux * probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
            temp3 = (data_aux - mus[cluster]) * probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (data_aux - mus[cluster]).T
            covs[cluster] = np.dot(temp4, temp3)/(probabilities.sum(axis=1)[cluster] + 1e-308) + self.__C(dataset, probabilities, cov_cluster, cluster)

        priors = probabilities.sum(axis=1)/self.n

        return priors, mus, covs, imputed_dataset

    def __impute_em(self, X, max_iter = 100, eps = 1e-8):
        '''(np.array, int, number) -> {str: np.array or int}

        Precondition: max_iter >= 1 and eps > 0

        Return the dictionary with five keys where:
        - Key 'mu' stores the mean estimate of the imputed data.
        - Key 'Sigma' stores the variance estimate of the imputed data.
        - Key 'X_imputed' stores the imputed data that is mutated from X using
          the EM algorithm.
        - Key 'C' stores the np.array that specifies the original missing entries
          of X.
        - Key 'iteration' stores the number of iteration used to compute
          'X_imputed' based on max_iter and eps specified.
        '''

        nr, nc = X.shape
        C = np.isnan(X) == False

        # Collect M_i and O_i's
        one_to_nc = np.arange(1, nc + 1, step = 1)
        M = one_to_nc * (C == False) - 1
        O = one_to_nc * C - 1

        # Generate Mu_0 and Sigma_0
        Mu = np.nanmean(X, axis = 0)
        observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]

        S = np.cov(X[observed_rows, ].T)
        if np.isnan(S).any():
            S = np.diag(np.nanvar(X, axis = 0))

        # Start updating
        Mu_tilde, S_tilde = {}, {}
        X_tilde = X.copy()
        no_conv = True
        iteration = 0
        while no_conv and iteration < max_iter:
            for i in range(nr):
                S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
                if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                    M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                    S_MM = S[np.ix_(M_i, M_i)]
                    S_MO = S[np.ix_(M_i, O_i)]
                    S_OM = S_MO.T
                    S_OO = S[np.ix_(O_i, O_i)]
                    Mu_tilde[i] = Mu[np.ix_(M_i)] +\
                        S_MO @ np.linalg.inv(S_OO + 1e-6 * np.identity(S_OO.shape[0])) @\
                        (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO + 1e-6 * np.identity(S_OO.shape[0])) @ S_OM
                    S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
            Mu_new = np.mean(X_tilde, axis = 0)
            S_new = np.cov(X_tilde.T, bias = 1) +\
                reduce(np.add, S_tilde.values()) / nr
            no_conv =\
                np.linalg.norm(Mu - Mu_new) >= eps or\
                np.linalg.norm(S - S_new, ord = 2) >= eps
            Mu = Mu_new
            S = S_new
            iteration += 1

        result = {
            'mu': Mu,
            'Sigma': S,
            'X_imputed': X_tilde,
            'C': C,
            'iteration': iteration
        }

        self.priors = np.array([1])
        self.mus = np.array([Mu])
        self.covs = np.array([S])

        return np.array([1]), np.array([Mu]), np.array([S]), X_tilde

    def impute(self, dataset, n_gaussians, n_iters=100, epsilon=1e-20, init='kmeans', verbose=False):
        self.n, self.d = dataset.shape
        self.n_gaussians = n_gaussians

        # if n_gaussians == 1:
        #     return self.__impute_em(dataset, max_iter=n_iters, eps=epsilon)

        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        mus = np.zeros((n_gaussians, self.d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)

        # cop = copy.deepcopy(dataset)
        # cop = np.where(np.isnan(cop), np.ma.array(cop, mask=np.isnan(cop)).mean(axis=0), cop)
        #
        # indices = np.array(np.where(np.all(~np.isnan(np.array(cop)), axis=1)))[0]
        # if init == 'kmeans' and len(indices) > 0:
        #     data_for_kmeans = cop[indices]
        #     kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)
        #     mus = kmeans.cluster_centers_
        # elif len(indices) > 0:
        #     for cluster in range(n_gaussians):
        #         mus[cluster] = dataset[indices[cluster]]
        # else:
        aux = np.nanmean(dataset, axis=0)
        for cluster in range(n_gaussians):
            mus[cluster] = aux

        for cluster in range(n_gaussians):
            # covs[cluster] = 1e-6 * np.identity(self.d)
            covs[cluster] = np.diag(np.nanvar(dataset, axis=0))


        for it in range(n_iters):
            mu_old = mus.copy()

            probabilities = self.__e_step(dataset, priors, mus, covs, n_gaussians)
            priors, mus, covs, imputed_dataset = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

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
