import sys
sys.path.append('..')

import math
import copy
import numpy as np
from scipy.spatial import distance

from methods.MICE import MICE
from methods.EM import EM
from methods.utils import *

from sklearn.metrics import mean_squared_error

# ************************ VARIABLES *******************************
missingness_percentage = [0.1, 0.3, 0.5]

##########################################
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split

# Not Mine!!
class MiceImputer(object):

    def __init__(self, seed_values = True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy # seed_strategy in ['mean','median','most_frequent', 'constant']
        self.seed_values = seed_values # seed_values = False initializes missing_values using not_null columns
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method = 'Linear', iter = 10, verbose = False):

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

                # if new_X[column].nunique() > 2:
                if method == 'Linear':
                    m = LinearRegression(n_jobs = -1)
                elif method == 'Ridge':
                    m = Ridge()
                m = LinearRegression()
                m.fit(train_x, train_y)
                model_score[i].append(m.score(val_x, val_y))
                new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                if verbose:
                    print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

                # elif new_X[column].nunique() == 2:
                #     if method == 'Linear':
                #         m = LogisticRegression(n_jobs = -1, solver = 'lbfgs')
                #     elif method == 'Ridge':
                #         m = RidgeClassifier()
                #     m = LogisticRegression(n_jobs=-1, solver='lbfgs')
                #     m.fit(train_x, train_y)
                #     model_score[i].append(m.score(val_x, val_y))
                #     new_X.loc[null_rows,column] = pd.Series(m.predict(test_x))
                #     if verbose:
                #         print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

            if model_score[i] == []:
                model_score[i] = 0
            else:
                model_score[i] = sum(model_score[i])/len(model_score[i])

        return new_X
##########################################

iterations = 500

for mis in missingness_percentage:
    print("MICE MISSINGNESS PERCENTAGE => ", mis)
    RMSE = 0

    for it in range(iterations):
        dataset = parse_file('mpg.csv')

        # dataset = (dataset - np.mean(dataset)) / np.std(dataset)
        dataset, dataset_missing = generate_missingness_instances(dataset, mis)

        # import pandas as pd
        # model = MiceImputer()
        # imputed = model.fit_transform(pd.DataFrame(copy.deepcopy(dataset_missing))).to_numpy()
        # print("MICE NOT MINE => ", math.sqrt(np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed)))

        model = MICE()
        imputed = model.impute(copy.deepcopy(dataset_missing))
        # print("MICE => ", math.sqrt(np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed)))
        RMSE += math.sqrt(np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed))

        # model = EM()
        # priors, mus, covs, imputed = model.impute(dataset_missing, 1)
        #
        # print("EM => ", math.sqrt(np.sum(abs(dataset - imputed) ** 2)/np.count_nonzero(dataset - imputed)))
        # print()
    print()
    print("RMSE => ", RMSE/iterations)
    print()
