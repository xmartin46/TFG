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
from numpy.random import normal
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from EM import EM

class ESD:
    def __init__(self):
        pass

    def estimateDistances(self, dataset, n_gaussians):
        """
        dataset has missing values
        """

        # Estimate the mean and covariance of the data set with the ECM algorithm (tornar a mirar)
        model = EM()
        priors, mus, covs, imputed_dataset = model.impute(dataset, n_gaussians, n_iters=50, epsilon=1e-8)

        nan_indices = np.where(np.any(np.isnan(dataset), axis=1))[0]
        variances = np.zeros((model.n), dtype=float)

        for indx in nan_indices:
            nan_Xm = np.isnan(dataset[indx])

            # cluster = 0
            # mu_m = model.mus[cluster][nan_Xm]
            # mu_o = model.mus[cluster][~nan_Xm]
            # cov_mo = model.covs[cluster][nan_Xm, :][:, ~nan_Xm]
            # cov_oo = model.covs[cluster][~nan_Xm, :][:, ~nan_Xm]
            # cov_oo_inverse = np.linalg.pinv(cov_oo)
            # aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (dataset[indx][~nan_Xm] - mu_o)[:,np.newaxis]))
            # nan_count = np.sum(nan_Xm)
            # dataset[indx, nan_Xm] = mu_m + aux.reshape(1, nan_count)


            aux = np.linalg.pinv(model.covs[0])
            cov_11 = np.linalg.pinv(aux[nan_Xm, :][:, nan_Xm])

            variances[indx] = np.trace(cov_11)

        P = np.zeros((model.n, model.n), dtype=float)

        for i in range(model.n):
            for j in range(model.n):
                Xi = imputed_dataset[i]
                Xj = imputed_dataset[j]

                for l in range(model.d):
                    P[i][j] += (Xi[l] - Xj[l]) ** 2

                P[i][j] += variances[i] + variances[j]

        return P
