import numpy as np

class IMean:
    def __init__(self):
        pass

    def impute(self, dataset):
        return np.where(np.isnan(dataset), np.ma.array(dataset, mask=np.isnan(dataset)).mean(axis=0), dataset)
