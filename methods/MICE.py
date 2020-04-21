import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier

from .utils import impute_methods

class MICE(impute_methods):
    def __init__(self):
        pass

    def impute(self, dataset, n_cycles=10):
        # Step 1: Impute missing values by the mean (it can be thought as a "place holder")
        mean = np.nanmean(dataset, axis=0)
        nan_indexes = np.isnan(dataset)

        for idx, i in enumerate(nan_indexes):
            dataset[idx, i] = mean[i]

        for _ in range(n_cycles):
            # For each attribute with missing data
            for idx in np.where(np.any(nan_indexes, axis=0))[0]:
                elem = dataset[:, idx]

                selector = [x for x in range(dataset.shape[1]) if x != idx]
                train_x = dataset[:, selector][~nan_indexes.T[idx]]
                train_y = elem[~nan_indexes.T[idx]]

                predict_x = dataset[:, selector][nan_indexes.T[idx]]

                # Observed values from the attribute are regressed on the other attributes in the
                # imputation model (consists of all other attributes in the model)
                model = LinearRegression()
                model.fit(train_x, train_y)

                elem[nan_indexes.T[idx]] = model.predict(predict_x)

        return dataset

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)
