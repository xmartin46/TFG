import random
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class RBFN:
    def __init__(self, k=2, lr=0.01, epochs=1000):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.w = None

    def __kmeans(self, X):
        kmeans = KMeans(n_clusters=self.k).fit(X.reshape(-1, 1)) ##############################################
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        std = []
        for c in range(self.k):
            cluster_list = []
            for indx, lx in enumerate(labels):
                if c == lx:
                    cluster_list.append(X[indx])
            std.append(np.std(cluster_list))

        return np.array(centers), np.array(std)

    def __gaussianRBF(self, x, c, r):
        return np.exp(-(distance.euclidean(x, c) ** 2)/(r ** 2))#, -2/(r ** 2) * np.exp(-(distance.euclidean(x, c) ** 2)/(r ** 2)) * distance.euclidean(x, c)

    def fit(self, X, Y):
        self.w = np.ones((self.k, Y.shape[1]), dtype=float)
        self.centroids, self.stds = self.__kmeans(X)

        # for epoch in range(self.epochs):
        #     e = 0
        #     for i in range(X.shape[0]):
        #         # forward pass
        #         a = np.array([self.__gaussianRBF(X[i], c, s) for c, s, in zip(self.centroids, self.stds)])
        #         F = a.T.dot(self.w)
        #
        #         # loss = (Y[i] - F).flatten() ** 2
        #         # print('Loss: {0:.2f}'.format(loss[0]))
        #
        #         # backward pass
        #         error = -(Y[i] - F).flatten()
        #         e += error
        #
        #         # online update
        #         self.w = self.w - self.lr * a * error
        print("sk", self.w)
        for epoch in range(self.epochs):
            HW = []
            DHI = []
            for i in range(X.shape[0]):
                hi = np.array([self.__gaussianRBF(X[i], c, s) for c, s, in zip(self.centroids, self.stds)])
                DHI.append(hi)
                HW.append(hi.T.dot(self.w))

            error = (HW - Y)
            self.w = self.w - self.lr * np.dot(np.array(DHI).T, error)
            print("Error: ", np.sum(error))
        print("sk", self.w)
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.__gaussianRBF(X[i], c, s) for c, s, in zip(self.centroids, self.stds)])
            F = a.T.dot(self.w)
            y_pred.append(F)
        return np.array(y_pred)

NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(2 * np.pi * X)  + noise

rbfnet = RBFN(lr=0.0035, k=2, epochs=500)
rbfnet.fit(X, np.array(y).reshape(NUM_SAMPLES, 1))

y_pred = rbfnet.predict(X)

plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()
