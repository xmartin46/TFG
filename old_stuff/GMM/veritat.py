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
from make_class import EM_estimate

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









# X1 = normal(loc=10, scale=2, size=10)
# X2 = normal(loc=20, scale=2, size=20)
# X3 = normal(loc=25, scale=3, size=30)
# X = hstack((X1, X2, X3))
# dataset = X.reshape((len(X), 1))
#
# plt.hist(X)
#
# k = 3
# pi, mu, sigma = EM(dataset, n_gaussians=k)
# print(pi)
# print()
# print(mu)
# print()
# print(sigma)
#
# for m in range(k):
#     t = 1000
#     x = np.linspace(0, 100, t)
#     plt.plot(x, 1000 * stats.norm.pdf(x, mu[m], sigma[m])[0])
# plt.show()





mean1 = [-20, -20]
cov1 = [[5, 0],
        [0, 5]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
plt.plot(x1, y1, 'x')

mean2 = [0, 0]
cov2 = [[50, -10],
        [-10, 50]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
plt.plot(x2, y2, 'x')

mean3 = [50, 50]
cov3 = [[10, 0],
        [0, 10]]
x3, y3 = np.random.multivariate_normal(mean3, cov3, 1000).T
plt.plot(x3, y3, 'x')


plt.axis('equal')
# plt.show()

d1 = list(zip(x1, y1))
d2 = list(zip(x2, y2))
d3 = list(zip(x3, y3))
dataset = d1 + d2 + d3
dataset = np.array(dataset)

model = EM_estimate()
priors, mus, covs = model.estimate(dataset, 3)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

print("PRIORS => ", priors)
print()
print("MUS:")
print(mus)
print()
print("COVS:")
print(covs)

x = [a[0] for a in mus]
y = [a[1] for a in mus]
plt.plot(x, y, 'r+')
plt.show()
