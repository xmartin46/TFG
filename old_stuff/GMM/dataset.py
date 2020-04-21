import numpy as np ,sys, time
from scipy.stats import multivariate_normal
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import hstack
from numpy.random import normal
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from sklearn.cluster import KMeans
import copy
from tqdm import tqdm

# def e_step(dataset, priors, mus, covs, n_gaussians):
#     n, d = dataset.shape
#     probabilities = np.zeros((n_gaussians, n), dtype=float)
#
#     for cluster in range(n_gaussians):
#         probabilities[cluster] = multivariate_normal.pdf(dataset, mean=mus[cluster], cov=covs[cluster], allow_singular=True)
#     aux = probabilities.sum(axis=0)
#
#     return probabilities/(aux + 1e-308)
#
# def m_step(dataset, probabilities, priors, mus, covs, n_gaussians):
#     n, d = dataset.shape
#
#     for cluster in range(n_gaussians):
#         # do = np.zeros((d, d), dtype=float)
#         # mus[cluster] = np.dot(probabilities[cluster], dataset)/sum(probabilities[cluster])
#         #
#         # for i in range(n):
#         #     z = np.array(dataset[i] - mus[cluster])
#         #     z = z.reshape(len(z), 1)
#         #     a = z * probabilities[cluster][i]
#         #     a = a.reshape(len(a), 1)
#         #     do += np.dot(a, z.T)
#         #
#         # covs[cluster] = do/probabilities.sum(axis=1)[cluster]
#
#         temp3 = (dataset -mus[cluster])*  probabilities.T[:,cluster][:,np.newaxis]
#         temp4 = (dataset-mus[cluster]).T
#         covs[cluster] = np.dot(temp4,temp3)  / probabilities.sum(axis=1)[cluster]
#         mus[cluster] = (dataset*  probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/ probabilities.sum(axis=1)[cluster]
#
#     priors = probabilities.sum(axis=1)/n
#     return priors, mus, covs
#
# def EM(dataset, n_gaussians, n_iters=200, epsilon=1e-20):
#     n, d = dataset.shape
#
#     # K-Means initialization
#     kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(dataset)
#     mus = kmeans.cluster_centers_
#
#     priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
#     # mus = np.zeros((n_gaussians, d), dtype=float)
#     covs = np.zeros((n_gaussians, d, d), dtype=float)
#
#     for cluster in range(n_gaussians):
#         # mus[cluster] = dataset[cluster]
#         covs[cluster] = np.identity(d)
#
#     for it in range(n_iters):
#         mu_old = mus.copy()
#         # probabilities size: n_gaussians x n
#         probabilities = e_step(dataset, priors, mus, covs, n_gaussians)
#
#         priors, mus, covs = m_step(dataset, probabilities, priors, mus, covs, n_gaussians)
#
#         #convergence?
#         temp = 0
#         for j in range(n_gaussians):
#             temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
#         temp = round(temp,20)
#         if temp <= epsilon:
#             print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))
#             break
#
#     return priors, mus, covs



def e_step(dataset, priors, mus, covs, n_gaussians):
    n, d = dataset.shape
    probabilities = np.zeros((n_gaussians, n), dtype=float)

    for cluster in range(n_gaussians):
        for i in range(n):
            mu_cluster = mus[cluster]
            cov_cluster = covs[cluster]

            nan_indexes = np.isnan(dataset[i])
            mu_o = mu_cluster[~nan_indexes]
            cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]

            # print("mu => ", mu_cluster)
            # print("dataset[i] => ", dataset[i])
            # print("x_o => ", dataset[i][~nan_indexes])
            # print("nan_indexes => ", nan_indexes)
            # print("mu_o: ", mu_o)
            # print("cov_oo: ", cov_oo)
            # print()
            # print()

            probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo, allow_singular=True)
    aux = probabilities.sum(axis=0)

    return probabilities/(aux + 1e-308)

def m_step(dataset, probabilities, priors, mus, covs, n_gaussians):
    n, d = dataset.shape

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

            aux = np.dot(cov_mo,
                            np.dot(np.linalg.pinv(cov_oo), (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
            nan_count = np.sum(nan_indexes)
            data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)

            if cluster == elem_belong_to_cluster[i]:
                imputed_dataset[i] = data_aux[i]

        mus[cluster] = (data_aux *  probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
        temp3 = (data_aux - mus[cluster])*  probabilities.T[:,cluster][:,np.newaxis]
        temp4 = (data_aux - mus[cluster]).T
        covs[cluster] = np.dot(temp4, temp3) /(probabilities.sum(axis=1)[cluster] + 1e-308)

    priors = probabilities.sum(axis=1)/n

    return priors, mus, covs, imputed_dataset

def EM(dataset, n_gaussians, n_iters=100, epsilon=1e-5):
    n, d = dataset.shape

    priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
    mus = np.zeros((n_gaussians, d), dtype=float)
    covs = np.zeros((n_gaussians, d, d), dtype=float)

    indices = np.array(np.where(np.all(~np.isnan(np.array(dataset)), axis=1)))[0]
    data_for_kmeans = dataset[indices]

    # K-Means initialization
    kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)
    mus = kmeans.cluster_centers_


    for cluster in range(n_gaussians):
        # mus[cluster] = dataset[indices[cluster]]
        covs[cluster] = np.identity(d)

    for it in tqdm(range(n_iters)):
        mu_old = mus.copy()

        # probabilities size: n_gaussians x n
        probabilities = e_step(dataset, priors, mus, covs, n_gaussians)

        priors, mus, covs, imputed_dataset = m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

        #convergence?
        temp = 0
        for j in range(n_gaussians):
            temp = temp + np.sqrt(np.power((mus[j] - mu_old[j]),2).sum())
        temp = round(temp,20)
        if temp <= epsilon:
            break
    print ("Iteration number = %d, stopping criterion = %.20f" %(it+1,temp))
    return priors, mus, covs, imputed_dataset
# ***********************************************************

class EMEM():
    """
    this algorithm just require to lean the Gauss distribution elements 'mu' and 'sigma'
    """
    def __init__(self,
                 max_iter=100,
                 theta=1e-5,
                 normalizer='min_max'):
        self.max_iter = max_iter
        self.theta = theta

    def _init_parameters(self, X):
        rows, cols = X.shape
        mu_init = np.nanmean(X, axis=0)
        sigma_init = np.zeros((cols, cols))
        for i in range(cols):
            for j in range(i, cols):
                vec_col = X[:, [i, j]]
                vec_col = vec_col[~np.any(np.isnan(vec_col), axis=1), :].T
                if len(vec_col) > 0:
                    cov = np.cov(vec_col)
                    cov = cov[0, 1]
                    sigma_init[i, j] = cov
                    sigma_init[j, i] = cov

                else:
                    sigma_init[i, j] = 1.0
                    sigma_init[j, i] = 1.0

        return mu_init, sigma_init

    def _e_step(self, mu,sigma, X):
        samples,_ = X.shape
        for sample in range(samples):
            if np.any(np.isnan(X[sample,:])):
                loc_nan = np.isnan(X[sample,:])
                new_mu = np.dot(sigma[loc_nan, :][:, ~loc_nan],
                                np.dot(np.linalg.inv(sigma[~loc_nan, :][:, ~loc_nan]),
                                       (X[sample, ~loc_nan] - mu[~loc_nan])[:,np.newaxis]))
                nan_count = np.sum(loc_nan)
                X[sample, loc_nan] = mu[loc_nan] + new_mu.reshape(1,nan_count)
        return X

    def _m_step(self,X):
        rows, cols = X.shape
        mu = np.mean(X, axis=0)
        sigma = np.cov(X.T)
        tmp_theta = -0.5 * rows * (cols * np.log(2 * np.pi) +
                                  np.log(np.linalg.det(sigma)))

        return mu, sigma,tmp_theta



    def solve(self, X):
        mu, sigma = self._init_parameters(X)
        complete_X,updated_X = None, None
        rows,_ = X.shape
        theta = -np.inf
        for iter in range(self.max_iter):
            updated_X = self._e_step(mu=mu, sigma=sigma, X=copy.copy(X))
            mu, sigma, tmp_theta = self._m_step(updated_X)
            for i in range(rows):
                tmp_theta -= 0.5 * np.dot((updated_X[i, :] - mu),
                                          np.dot(np.linalg.inv(sigma), (updated_X[i, :] - mu)[:, np.newaxis]))
            if abs(tmp_theta-theta)<self.theta:
                complete_X = updated_X
                break;
            else:
                theta = tmp_theta
        else:
            complete_X = updated_X

        return complete_X

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

    print(len(dataset))
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

# GENERATE DATASETS
n = 500
d = 15
#   AR
rho = 0.9

missingness_percentage = 0.1

dataset = generate_dataset_AR(n, d, rho)
# dataset1 = generate_dataset_AR(n, d, 0.9)
# dataset = np.concatenate((dataset, dataset1), axis=0)
dataset, dataset_missing = generate_missingness(dataset, missingness_percentage)

n, _ = dataset_missing.shape
print()
print(n)
print()

priors, mus, covs, imputed_dataset = EM(dataset_missing, 2)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

print("ERROR:")
print(np.sum(abs(dataset - imputed_dataset) ** 2)/np.count_nonzero(dataset - imputed_dataset))


model = EMEM()
completeX = model.solve(dataset_missing)
print("ERROR SEU:")
print(np.sum(abs(dataset - completeX) ** 2)/np.count_nonzero(dataset - completeX))
