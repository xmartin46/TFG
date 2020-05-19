import sys
sys.path.append('..')

import time

from methods.utils import *

from methods.MICE import MICE
from methods.kNN import kNN, wNN, wNN_correlation
from methods.EM import EM
from methods.IMean import IMean
from methods.IZero import IZero
from methods.ESD import ESD
from methods.EED import EED

from tqdm import tqdm

# 50 iteracions
# Guardem RMSE de totes les iteracions (test de significancia)
# Anem augmentat el missing rate (MAR)
# Hem de trobar els millors paràmetres de cada mètode per cada dataset
# Normalitzem dades????

iterations = 10
dataset_name = 'wine.csv'
missingness_percentages = [0.1] #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dataset_attributes = [[100, 30, 0.9]] #, [100, 30, 0.5], [100, 30, 0.1]] #[n, d, rho]

lambds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
neighborss = [5, 10, 15, 20, 25, 30]
ms = range(1, 8)

for n, d, rho in dataset_attributes:
    start = time.time()
    bsf = float("inf")
    bl = 0
    bn = 0
    bm = 0
    for mis in missingness_percentages:
        for lambd in lambds:
            for neighbors in neighborss:
                for m in ms:
                    print([n, d, rho], " => LAMBDA ", lambd, " |  neighbors ", neighbors, " |   m ", m)
                    MSIEkNN = 0
                    MSIEwNN = 0
                    MSIEwNN_correlation = 0

                    for it in range(iterations):
                        # dataset = parse_file(dataset_name)
                        dataset = generate_dataset_AR(n, d, rho)

                        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
                        dataset, dataset_missing = generate_missingness_MAR(dataset, mis)



                        # # kNN
                        # methodkNN = kNN()
                        # q = 2
                        # imputed_kNN = methodkNN.impute(copy.deepcopy(dataset_missing), neighbors, q)
                        # MSIEkNN += methodkNN.MSIE(dataset, imputed_kNN)
                        # # print("kNN FINISHED")


                        # # wNN
                        # methodwNN = wNN()
                        # q = 2
                        # kernel_type = 'Gaussian'
                        # imputedwNN = methodwNN.impute(copy.deepcopy(dataset_missing), neighbors, q, kernel_type, lambd)
                        # MSIEwNN += methodwNN.MSIE(dataset, imputedwNN)
                        # # print("wNN FINISHED")


                        # wNN_correlation (Els que els hi falti valors des del principi, avisar que primer fem un knn imputant la mitjana dels neighbors)
                        methodwNN_correlation = wNN_correlation()
                        q = 2
                        kernel_type = 'Gaussian'
                        imputedwNN_correlation = methodwNN_correlation.impute(dataset, copy.deepcopy(dataset_missing), neighbors, q, kernel_type, lambd, m)
                        MSIEwNN_correlation += methodwNN_correlation.MSIE(dataset, imputedwNN_correlation)
                        # print("wNN_correlation FINISHED")

                    # print("kNN", MSIEkNN/iterations)
                    # print("wNN", MSIEwNN/iterations)
                    print("wNN_correlation", MSIEwNN_correlation/iterations)

                    if MSIEwNN_correlation/iterations < bsf:
                        bsf = MSIEwNN_correlation/iterations
                        bl = lambd
                        bn = neighbors
                        bm = m
                        print("New!")

    np.save(f"./data/AR/tunning/dataset_n_{n}_d_{d}_rho_{rho}_parameters", np.array([bl, bn, bm]), allow_pickle=True)
    print(np.load(f"./data/AR/tunning/dataset_n_{n}_d_{d}_rho_{rho}_parameters.npy", allow_pickle=True))
    print(bl)
    print(bn)
    print(bm)
    print("Time => ", time.time() - start)
