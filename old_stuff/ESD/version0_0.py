# Distance estimation in numerical data sets with missing values
# Authors: Emil Eirola, Gauthier Doquire, Michael Verleysen, Amaury Lendasse

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

from sklearn import datasets

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

def ESD(dataset, n_gaussians):
    """
    dataset has missing values
    """

    # Estimate the mean and covariance of the data set with the ECM algorithm (tornar a mirar)
    model = EM()
    priors, mus, covs, _ = model.impute(dataset, n_gaussians)

    nan_indices = np.where(np.any(np.isnan(dataset), axis=1))[0]
    variances = np.zeros((model.n), dtype=float)
    cluster = 0
    for indx in nan_indices:
        nan_Xm = np.isnan(dataset[indx])

        mu_m = model.mus[cluster][nan_Xm]
        mu_o = model.mus[cluster][~nan_Xm]
        cov_mo = model.covs[cluster][nan_Xm, :][:, ~nan_Xm]
        cov_oo = model.covs[cluster][~nan_Xm, :][:, ~nan_Xm]
        cov_oo_inverse = np.linalg.pinv(cov_oo)

        aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (dataset[indx][~nan_Xm] - mu_o)[:,np.newaxis]))
        nan_count = np.sum(nan_Xm)
        dataset[indx, nan_Xm] = mu_m + aux.reshape(1, nan_count)


        aux = np.linalg.pinv(model.covs[0])
        cov_11 = np.linalg.pinv(aux[nan_Xm, :][:, nan_Xm])

        variances[indx] = np.trace(cov_11)

    P = np.zeros((model.n, model.n), dtype=float)

    for i in range(model.n):
        for j in range(model.n):
            Xi = dataset[i]
            Xj = dataset[j]

            s = 0
            for l in range(model.d):
                s += (Xi[l] - Xj[l]) ** 2

            P[i][j] = s + variances[i] + variances[j]

    return P

def generate_missingness(dataset, missingness_percentage):
    n, d = dataset.shape
    dataset_real = copy.deepcopy(dataset)
    # dataset = dataset.flatten()

    # L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    # for j in range(len(L)):
    #     dataset[L[j]] = float('nan')
    #
    # dataset = np.array(dataset)
    # dataset = np.split(dataset, n)
    # dataset = dataset

    for i in range(n):
        for j in range(d):
            if random.random() < missingness_percentage:
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

def realDistances(dataset):
    n, d = dataset.shape

    P = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            Xi = dataset[i]
            Xj = dataset[j]

            s = 0
            for l in range(d):
                s += (Xi[l] - Xj[l]) ** 2

            P[i][j] = s

    return P

missingness_percentage = [0.05, 0.15, 0.3, 0.6]
iterations = 1

from ecoli import ecoli
from wine import wine

for mis in missingness_percentage:
    print(f"MISSINGNESS PERCENTAGE => {mis}")
    RMSE = 0
    for it in tqdm(range(iterations)):

        iris = datasets.load_iris()
        dataset = iris.data
        # dataset = ecoli
        # dataset = wine
        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_missingness(copy.deepcopy(dataset), mis)

        n, d = dataset_missing.shape
        # print(dataset_missing.shape)

        P = ESD(copy.deepcopy(dataset_missing), 1)
        R = realDistances(dataset)

        s = 0
        nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
        for i in nan_indices:
            for j in nan_indices:
                if i > j:
                    s += (math.sqrt(P[i][j]) - math.sqrt(R[i][j])) ** 2

        count = len(nan_indices)
        RMSE += math.sqrt(s/(count * n - count * (count + 1)/2))
        print(math.sqrt(s/(count * n - count * (count + 1)/2)))

    print(f"RMSE => {RMSE/iterations}")
    print()
