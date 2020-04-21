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
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class EM:
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

                aux = np.linalg.pinv(cov)
                aux = np.linalg.pinv(cov[nan_indexes, :][:, nan_indexes])

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

                probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo, allow_singular=True)

        aux = probabilities.sum(axis=0)

        return probabilities/(aux + 1e-308)

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):
        elem_belong_to_cluster = np.argmax(probabilities, axis=0)

        imputed_dataset = copy.deepcopy(dataset)

        for cluster in range(n_gaussians):
            # Expect missing values
            data_aux = copy.deepcopy(dataset)
            for i in range(n):
                mu_cluster = mus[cluster]
                cov_cluster = covs[cluster]

                nan_indexes = np.isnan(dataset[i])
                mu_m = mu_cluster[nan_indexes]
                mu_o = mu_cluster[~nan_indexes]
                cov_mo = cov_cluster[nan_indexes, :][:, ~nan_indexes]
                cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                cov_oo_inverse = np.linalg.pinv(cov_oo)

                aux = np.dot(cov_mo,
                                np.dot(cov_oo_inverse, (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
                nan_count = np.sum(nan_indexes)
                data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)

                if cluster == elem_belong_to_cluster[i]:
                    imputed_dataset[i] = data_aux[i]

            mus[cluster] = (data_aux *  probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
            temp3 = (data_aux - mus[cluster])*  probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (data_aux - mus[cluster]).T
            covs[cluster] = (np.dot(temp4, temp3) + self.__C(dataset, probabilities, cov_cluster, cluster))/(probabilities.sum(axis=1)[cluster] + 1e-308)

        priors = probabilities.sum(axis=1)/n

        return priors, mus, covs, imputed_dataset

    def impute(self, dataset, n_gaussians, n_iters=100, epsilon=1e-200, init='kmeans'):
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

        # print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))

        self.priors = priors
        self.mus = mus
        self.covs = covs

        return priors, mus, covs, imputed_dataset

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




def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def generate_missingness(dataset, missingness_percentage):
    n, d = dataset.shape
    dataset_real = copy.copy(dataset)
    dataset = dataset.flatten()

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = dataset

    # print(len(dataset))

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(dataset)), axis=1)))[0]
    indices.sort()
    indices = np.flip(indices)

    for i in indices:
        dataset.pop(i)
        dataset_real = np.delete(dataset_real, i, axis=0)

    return dataset_real, np.array(dataset)

def gen_dataset():
    mean1 = [-20, -20]
    cov1 = [[5, 0],
            [0, 5]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
    plt.plot(x1, y1, 'x')

    mean2 = [0, 0]
    cov2 = [[50, -10],
            [-10, 50]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
    plt.plot(x2, y2, 'x')

    mean3 = [50, 50]
    cov3 = [[10, 0],
            [0, 10]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 100).T
    # plt.plot(x3, y3, 'x')
    plt.axis('equal')
    # plt.show()

    d1 = list(zip(x1, y1))
    d2 = list(zip(x2, y2))
    d3 = list(zip(x3, y3))
    dataset = d1 + d2
    dataset = np.array(dataset)

    return dataset

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

real_and_not_real = 0
real_and_imputed = 0

missingness_percentage = 0.1

print("MISSINGNESS_PERCENTAGE => ", missingness_percentage)

for it in range(1, 10):
    # GENERATE DATASETS
    n = 100
    d = 30
    #   AR
    rho = 0.9

    # missingness_percentage = 0.2

    mean = np.array([-0.3, 0.1, 2])
    cov = np.array([[0.4, 0.15, 0.25], [0.15, 0.25, 0.1], [0.25, 0.1, 0.3]])
    dataset = np.random.multivariate_normal(mean, cov, n)

    # dataset = generate_dataset_AR(n, d, rho)
    # dataset = (dataset - np.mean(dataset)) / np.std(dataset)
    dataset, dataset_missing = generate_missingness(dataset, missingness_percentage)

    n, d = dataset_missing.shape

    model = EM()
    priors, mus, covs, imputed_dataset = model.impute(dataset_missing, 1)
    # print(priors)
    # print()
    # print(mus)
    # print()
    # print(covs)
    # print()
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    # print("ERROR:")
    # print(np.sum(abs(dataset - imputed_dataset) ** 2)/np.count_nonzero(dataset - imputed_dataset))

    # *****************************************************************************
    indices = np.array(np.where(np.any(np.isnan(dataset_missing), axis=1)))[0]

    real = []
    imputed = []
    not_real = []

    for i, _ in enumerate(indices):
        for j, _ in enumerate(indices):
            if i != j:
                not_real.append(EED(dataset_missing[indices[i]], dataset_missing[indices[j]], model))
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
