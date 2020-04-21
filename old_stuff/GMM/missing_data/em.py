import numpy as np
import copy
import random
import math

class EM():
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

    dataset = dataset.flatten()

    L = random.sample(range(n * d), math.floor(d * n * missingness_percentage))
    for j in range(len(L)):
        dataset[L[j]] = float('nan')

    dataset = np.array(dataset)
    dataset = np.split(dataset, n)
    dataset = np.array(dataset)

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
# *******************************************************************

# *************************** VARIABLES *****************************
# REPETITIONS
samples = 1

# GENERATE DATASETS
n = 100
d = 5
#   AR
rho = 0.9

# MISSINGNESS
missingness_percentage = 0.05

# ******************************************************************

for _ in range(samples):
    datasetAR = generate_dataset_AR(n, d, rho)

    datasetAR_missing = generate_missingness(datasetAR, missingness_percentage)
    print(datasetAR_missing)
    print()
    print()

    model = EM()
    completeX = model.solve(datasetAR_missing)

    print(datasetAR)
    print()
    print()
    print(completeX)
    print()
    print()
    print(datasetAR - completeX)

    print(np.sum(abs(datasetAR - completeX) ** 2)/np.count_nonzero(datasetAR - completeX))
