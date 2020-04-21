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

            temp3 = (dataset -mus[cluster])*  probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (dataset-mus[cluster]).T
            covs[cluster] = np.dot(temp4,temp3)  / probabilities.sum(axis=1)[cluster]
            mus[cluster] = (dataset*  probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/ probabilities.sum(axis=1)[cluster]

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

class EM(impute_methods):
    def __init__(self):
        self.n = None
        self.d = None

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

    def impute(self, dataset, n_gaussians, n_iters=100, epsilon=1e-20, init='kmeans'):
        self.n, self.d = dataset.shape
        print(dataset.shape)

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




        for it in tqdm(range(n_iters)):
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

        print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))

        return priors, mus, covs, imputed_dataset

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
    dataset_real = copy.copy(dataset)
    dataset = dataset.flatten()

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = dataset

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(dataset)), axis=1)))[0]
    indices.sort()
    indices = np.flip(indices)

    for i in indices:
        dataset.pop(i)
        dataset_real = np.delete(dataset_real, i, axis=0)

    return dataset_real, np.array(dataset)

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
missingness_percentage = 0.5

# NEAREST NEIGHBOR
neighbors = 10
q = 2
lambd = 1
mC = 6
kernels = ['Gaussian',
           'Tricube']
kernel_type = kernels[0]
# ******************************************************************

model = EM()

for _ in tqdm(range(samples)):
    datasetAR = generate_dataset_AR(n, d, rho)
    datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)

    datasetAR, datasetAR_missing = generate_missingness(datasetAR, missingness_percentage)
    datasetBW, datasetBW_missing = generate_missingness(datasetBW, missingness_percentage)

    priors, mus, covs, imputed_dataset = model.impute(datasetAR_missing, 1)

    print("MSIE AR => ", model.MSIE(datasetAR, imputed_dataset))
    print("MAIE AR => ", model.MAIE(datasetAR, imputed_dataset))


    priors, mus, covs, imputed_dataset = model.impute(datasetBW_missing, 1)

    print("MSIE BW => ", model.MSIE(datasetBW, imputed_dataset))
    print("MAIE BW => ", model.MAIE(datasetBW, imputed_dataset))
