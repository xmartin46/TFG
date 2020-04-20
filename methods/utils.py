import copy
import math
import random
import warnings
import numpy as np
from numpy.random import normal
from sklearn.preprocessing import normalize

class impute_methods:
    def __init__(self):
        pass

    def MSIE(self, real, imputed):
        return np.sum(abs(real - imputed) ** 2)/np.count_nonzero(real - imputed)

    def MAIE(self, real, imputed):
        return np.sum(abs(real - imputed))/np.count_nonzero(real - imputed)

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

# *************************************** MISSINGNESS **********************************************
# https://rmisstastic.netlify.com/how-to/python/generate_html/how%20to%20generate%20missing%20values
# Amputation
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

# MCAR
def generate_MCAR(dataset, missingness_percentage):
    # Mask completly at random some values

    M = np.random.binomial(1, missingness_percentage, size = dataset.shape)
    X_obs = dataset.copy()
    np.putmask(X_obs, M, np.nan)
    print('Percentage of newly generated mising values (MCAR): {}'.\
      format(np.round(np.sum(np.isnan(X_obs))/X_obs.size,3)))

    # # warning if a full row is missing
    # for row in X_obs:
    #     if np.all(np.isnan(row)):
    #         warnings.warn('Some row(s) contains only nan values.')
    #         break
    #
    # # warning if a full col is missing
    # for col in X_obs.T:
    #     if np.all(np.isnan(col)):
    #         warnings.warn('Some col(s) contains only nan values.')
    #         break

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(X_obs)), axis=1)))[0]
    indices.sort()
    indices = np.flip(indices)

    X_obs = np.delete(X_obs, indices, 0)
    dataset = np.delete(dataset, indices, 0)

    return dataset, X_obs

# MAR
def generate_MAR(dataset, missingness_percentage, W=None):
    """ Observed values will censor the missing ones

    The proba of being missing: M_proba = X_obs.dot(W)
    So for each sample, some observed feature (P=1) will influence
    the missingness of some others features (P=0) w.r.t to the weight
    matrix W (shape n_features x n_features).

    e.g. during a questionnary, those who said being busy (X_obs[:,0] = 1)
    usualy miss to fill the last question (X_obs[:,-1] = np.nan)
    So here W[0,-1] = 1
    """
    X_obs = dataset.copy()
    M_proba = np.zeros(X_obs.shape)

    if W is None:
        # generate the weigth matrix W
        W = np.random.randn(dataset.shape[1], dataset.shape[1])

    # Severals iteration to have room for high missing_rate
    for i in range(X_obs.shape[1]*2):
        # Sample a pattern matrix P
        # P[i,j] = 1 will correspond to an observed value
        # P[i,j] = 0 will correspond to a potential missing value
        P = np.random.binomial(1, .5, size=dataset.shape)

        # potential missing entry do not take part of missingness computation
        X_not_missing = np.multiply(dataset, P)

        # sample from the proba X_obs.dot(W)
        sigma = np.var(X_not_missing)
        M_proba_ = np.random.normal(X_not_missing.dot(W), scale = sigma)

        # not missing should have M_proba = 0
        M_proba_ = np.multiply(M_proba_, 1-P)  # M_proba[P] = 0

        M_proba += M_proba_

    thresold = np.percentile(M_proba.ravel(), 100 * (1 - missingness_percentage))
    M = M_proba > thresold

    np.putmask(X_obs, M, np.nan)
    print('Percentage of newly generated mising values (MAR): {}'.\
      format(np.sum(np.isnan(X_obs))/X_obs.size))

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(X_obs)), axis=1)))[0]
    indices.sort()
    indices = np.flip(indices)

    X_obs = np.delete(X_obs, indices, 0)
    dataset = np.delete(dataset, indices, 0)

    return dataset, X_obs

# NMAR
def generate_NMAR(dataset, missingness_percentage):
    """" ampute X_complete with censoring (Missing Not At Random)

    The missingness depends on the values.
    This will tends to "censor" X[i,j] where X[i,j] is high
    comparing to its column X[:,j]
    """

    # M depends on X_complete values
    M_proba = np.random.normal(dataset)
    M_proba = normalize(M_proba, norm='l1')

    # compute thresold wrt missing_rate
    thresold = np.percentile(M_proba.ravel(), 100 * (1 - missingness_percentage))
    M = M_proba > thresold

    X_obs = dataset.copy()
    np.putmask(X_obs, M, np.nan)
    print('Percentage of newly generated mising values (NMAR): {}'.\
      format(np.sum(np.isnan(X_obs))/X_obs.size))

    # Delete items with all NaN
    indices = np.array(np.where(np.all(np.isnan(np.array(X_obs)), axis=1)))[0]
    indices.sort()
    indices = np.flip(indices)

    X_obs = np.delete(X_obs, indices, 0)
    dataset = np.delete(dataset, indices, 0)

    return dataset, X_obs
