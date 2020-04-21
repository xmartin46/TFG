# -*- coding: utf-8 -*-
"""
Multiple Imputation by Chained Equations
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
"""

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

import pandas as pd
import numpy as np

from make_class import MICE

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split

class MiceImputer(object):

    def __init__(self, seed_values = True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy # seed_strategy in ['mean','median','most_frequent', 'constant']
        self.seed_values = seed_values # seed_values = False initializes missing_values using not_null columns
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method = 'Linear', iter = 10, verbose = True):

        # Why use Pandas?
        # http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
        # Pandas < Numpy if X.shape[0] < 50K
        # Pandas > Numpy if X.shape[0] > 500K

        # Data necessary for masking missing-values after imputation
        null_cols = X.columns[X.isna().any()].tolist()
        null_X = X.isna()[null_cols]

        ### Initialize missing_values

        if self.seed_values:

            # Impute all missing values using SimpleImputer
            if verbose:
                print('Initilization of missing-values using SimpleImputer')
            new_X = pd.DataFrame(self.imp.fit_transform(X))
            new_X.columns = X.columns
            new_X.index = X.index


        ### Begin iterations of MICE

        model_score = {}

        for i in range(iter):
            if verbose:
                print('Beginning iteration ' + str(i) + ':')

            model_score[i] = []

            for column in null_cols:

                null_rows = null_X[column]
                not_null_y = new_X.loc[~null_rows, column]
                not_null_X = new_X[~null_rows].drop(column, axis = 1)

                train_x, val_x, train_y, val_y = train_test_split(not_null_X, not_null_y, test_size=0.33, random_state=42)
                test_x = new_X.drop(column, axis = 1)

                if new_X[column].nunique() > 2:
                    if method == 'Linear':
                        m = LinearRegression(n_jobs = -1)
                    elif method == 'Ridge':
                        m = Ridge()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

                elif new_X[column].nunique() == 2:
                    if method == 'Linear':
                        m = LogisticRegression(n_jobs = -1, solver = 'lbfgs')
                    elif method == 'Ridge':
                        m = RidgeClassifier()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

            if model_score[i] == []:
                model_score[i] = 0
            else:
                model_score[i] = sum(model_score[i])/len(model_score[i])

        return new_X

class impute_methods:
    def __init__(self):
        pass

    def MSIE(self, real, imputed):
        return np.sum(abs(real - imputed) ** 2)/np.count_nonzero(real - imputed)

    def MAIE(self, real, imputed):
        return np.sum(abs(real - imputed))/np.count_nonzero(real - imputed)

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
            for i in range(self.n):
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

model = MiceImputer()

# MISSINGNESS
missingness_percentage = 0.15

# ******************************************************************

for _ in range(1):
    n = 2000
    d = 5
    mean = [random.random() * 10 for _ in range(d)]
    cov = np.random.rand(d, d) * 10
    cov = np.dot(cov, cov.T)

    dataset = np.random.multivariate_normal(mean, cov, n)
    dataset = (dataset - np.mean(dataset)) / np.std(dataset)

    dataset, dataset_missing = generate_missingness(dataset, missingness_percentage)

    imputed = model.fit_transform(copy.deepcopy(pd.DataFrame(dataset_missing)))
    imputed = imputed.to_numpy()
    print("MICE SEU => ", np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed))

    print()

    imputed = MICE().impute(copy.deepcopy(dataset_missing), n_cycles=10)
    print("MICE MEU => ", np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed))

    print()

    model = EM()
    priors, mus, covs, imputed_dataset = model.impute(dataset_missing, 1)

    print("EM => ", model.MSIE(dataset, imputed_dataset))
