import os

import sys
sys.path.append('..')

from methods.utils import *

from methods.MICE import MICE
from methods.kNN import kNN, wNN, wNN_correlation
from methods.EM import EM
from methods.IMean import IMean
from methods.IZero import IZero
from methods.ESD import ESD
from methods.EED import EED

import time
from tqdm import tqdm

# 50 iteracions
# Guardem RMSE de totes les iteracions (test de significancia)
# Anem augmentat el missing rate (MAR)
# Hem de trobar els millors paràmetres de cada mètode per cada dataset
# Normalitzem dades????

iterations = 30
dataset_name = 'wine.csv'
missingness_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
dataset_attributes = [[100, 30, 0.9]] #[n, d, rho]

for n, d, rho in dataset_attributes:
    ALL_ERRORS = []
    os.mkdir(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}")
    os.mkdir(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}/results")

    for mis in missingness_percentages:
        errorsEM = []
        errorsIMean = []
        errorsIZero = []
        errorskNN = []
        errorswNN = []
        errorswNN_correlation = []
        errorsESD = []
        errorsEED = []

        errors_in_one_miss = []
        start = time.time()
        for it in tqdm(range(iterations)):
            # dataset = parse_file(dataset_name)
            dataset = generate_dataset_AR(n, d, rho)

            dataset = (dataset - np.mean(dataset)) / np.std(dataset)
            dataset, dataset_missing = generate_missingness_MAR(dataset, mis)

            # Save files, just in case
            np.save(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}/dataset_mis_{mis}_it_{it}", dataset)
            np.save(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}/datasetmissing_mis_{mis}_it_{it}", dataset_missing)

            # EM (BIC selection up to 10 components)
            methodEM = EM()
            bic = BIC(dataset)
            print(bic)
            bic = 1
            _, _, _, imputedEM = methodEM.impute(copy.deepcopy(dataset_missing), bic, verbose=True)
            total_missing_values = np.count_nonzero(abs(dataset - imputedEM))
            errorEM = math.sqrt(np.sum(abs(dataset - imputedEM) ** 2)/total_missing_values)
            errorsEM.append(errorEM)
            print("EM FINISHED")


            # IMean
            methodIMean = IMean()
            imputedIMean = methodIMean.impute(copy.deepcopy(dataset_missing))
            total_missing_values = np.count_nonzero(abs(dataset - imputedIMean))
            errorIMean = math.sqrt(np.sum(abs(dataset - imputedIMean) ** 2)/total_missing_values)
            errorsIMean.append(errorIMean)
            print("IMean FINISHED")


            # IZero
            methodIZero = IZero()
            imputedIZero = methodIZero.impute(copy.deepcopy(dataset_missing))
            total_missing_values = np.count_nonzero(abs(dataset - imputedIZero))
            errorIZero = math.sqrt(np.sum(abs(dataset - imputedIZero) ** 2)/total_missing_values)
            errorsIZero.append(errorIZero)
            print("IZero FINISHED")


            # # kNN
            # methodkNN = kNN()
            # neighbors = 15
            # q = 2
            # imputed_kNN = methodkNN.impute(copy.deepcopy(dataset_missing), neighbors, q)
            # total_missing_values = np.count_nonzero(abs(dataset - imputed_kNN))
            # errorkNN = math.sqrt(np.sum(abs(dataset - imputed_kNN) ** 2)/total_missing_values)
            # errorskNN.append(errorkNN)
            # print("kNN FINISHED")
            #
            #
            # # wNN
            # methodwNN = wNN()
            # neighbors = 15
            # q = 2
            # kernel_type = 'Gaussian'
            # lambd = 0.1
            # imputedwNN = methodwNN.impute(copy.deepcopy(dataset_missing), neighbors, q, kernel_type, lambd)
            # total_missing_values = np.count_nonzero(abs(dataset - imputedwNN))
            # errorwNN = math.sqrt(np.sum(abs(dataset - imputedwNN) ** 2)/total_missing_values)
            # errorswNN.append(errorwNN)
            # print("wNN FINISHED")


            # wNN_correlation (Els que els hi falti valors des del principi, avisar que primer fem un knn imputant la mitjana dels neighbors)
            methodwNN_correlation = wNN_correlation()
            neighbors = 15
            q = 2
            kernel_type = 'Gaussian'
            lambd = 0.1
            m = 6
            imputedwNN_correlation = methodwNN_correlation.impute(dataset, copy.deepcopy(dataset_missing), neighbors, q, kernel_type, lambd, m)
            total_missing_values = np.count_nonzero(abs(dataset - imputedwNN_correlation))
            errorwNN_correlation = math.sqrt(np.sum(abs(dataset - imputedwNN_correlation) ** 2)/total_missing_values)
            errorswNN_correlation.append(errorwNN_correlation)
            print("wNN_correlation FINISHED")


            # expecting distances
            R = realDistances(dataset) # squared distances


            # ESD
            methodESD = ESD()
            P = methodESD.estimateDistances(copy.deepcopy(dataset_missing), bic)

            s = 0
            nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
            for i in range(dataset.shape[0]):
                for j in range(dataset.shape[0]):
                    if i > j:
                        if P[i][j] < 0:
                            print(P[i][j])
                        s += (math.sqrt(P[i][j]) - math.sqrt(R[i][j])) ** 2
            count = len(nan_indices)
            errorESD = math.sqrt(s/(count * dataset.shape[0] - count * (count + 1)/2))
            errorsESD.append(errorESD)
            print("ESD FINISHED")


            # EED
            methodEED = EED()
            P = methodEED.estimateDistances(copy.deepcopy(dataset_missing), bic)

            s = 0
            nan_indices = np.where(np.any(np.isnan(dataset_missing), axis=1))[0]
            for i in range(dataset.shape[0]):
                for j in range(dataset.shape[0]):
                    if i > j:
                        s += (P[i][j] - math.sqrt(R[i][j])) ** 2
            count = len(nan_indices)
            errorEED = math.sqrt(s/(count * dataset.shape[0] - count * (count + 1)/2))
            errorsEED.append(errorEED)
            print("EED FINISHED")


        print(time.time() - start)

        errors_in_one_miss.append(('EM', errorsEM))
        errors_in_one_miss.append(('IZero', errorsIZero))
        errors_in_one_miss.append(('IMean', errorsIMean))
        errors_in_one_miss.append(('kNN', errorskNN))
        errors_in_one_miss.append(('wNN', errorswNN))
        errors_in_one_miss.append(('wNN_correlation', errorswNN_correlation))
        errors_in_one_miss.append(('ESD', errorsESD))
        errors_in_one_miss.append(('EED', errorsEED))

        ALL_ERRORS.append((f'mis: {mis}', errors_in_one_miss))
        print(ALL_ERRORS)
        np.save(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}/results/ALL_ERRORS_n_{n}_d_{d}_rho_{rho}", ALL_ERRORS, allow_pickle=True)

    print(np.load(f"./data/AR/dataset_n_{n}_d_{d}_rho_{rho}/results/ALL_ERRORS_n_{n}_d_{d}_rho_{rho}.npy", allow_pickle=True))
