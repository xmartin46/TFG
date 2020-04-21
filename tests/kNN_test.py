import sys
sys.path.append('..')

import math
import copy
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

from methods.kNN import kNN, wNN, wNN_correlation
from methods.utils import *

# ************************ VARIABLES *******************************
samples = 1

# GENERATE DATASETS
n = 100
d = 15
#   AR
rho = 0.9
#   BW
rho_w = 0.9
rho_b = 0.1
predictors_per_block = 5

# MISSINGNESS
missingness_percentage = 0.15

# NEAREST NEIGHBOR
neighbors = 10
q = 2
lambd = 1
mC = 6
kernels = ['Gaussian',
           'Tricube']
kernel_type = kernels[0]
# ******************************************************************



MSIE_AR_kNN = 0
MAIE_AR_kNN = 0

MSIE_BW_kNN = 0
MAIE_BW_kNN = 0


MSIE_AR_wNN = 0
MAIE_AR_wNN = 0

MSIE_BW_wNN = 0
MAIE_BW_wNN = 0


MSIE_AR_wNNC = 0
MAIE_AR_wNNC = 0

MSIE_BW_wNNC = 0
MAIE_BW_wNNC = 0


kNN = kNN()
wNN = wNN()
wNNC = wNN_correlation()

for _ in tqdm(range(samples)):
    datasetAR = generate_dataset_AR(n, d, rho)
    datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)

    datasetAR, datasetAR_missing = generate_missingness_flatten(datasetAR, missingness_percentage)
    datasetBW, datasetBW_missing = generate_missingness_flatten(datasetBW, missingness_percentage)

    verify_dataset(datasetAR_missing)
    verify_dataset(datasetBW_missing)



    imputed_kNN_AR = kNN.impute(datasetAR_missing, neighbors, q)
    imputed_kNN_BW = kNN.impute(datasetBW_missing, neighbors, q)

    imputed_wNN_AR = wNN.impute(datasetAR_missing, neighbors, q, kernel_type, lambd)
    imputed_wNN_BW = wNN.impute(datasetBW_missing, neighbors, q, kernel_type, lambd)

    imputed_wNNC_AR = wNNC.impute(datasetAR, datasetAR_missing, neighbors, q, kernel_type, lambd)
    imputed_wNNC_BW = wNNC.impute(datasetBW, datasetBW_missing, neighbors, q, kernel_type, lambd)



    MSIE_AR_kNN += kNN.MSIE(datasetAR, imputed_kNN_AR)
    MAIE_AR_kNN += kNN.MAIE(datasetAR, imputed_kNN_AR)

    MSIE_BW_kNN += kNN.MSIE(datasetBW, imputed_kNN_BW)
    MAIE_BW_kNN += kNN.MAIE(datasetBW, imputed_kNN_BW)


    MSIE_AR_wNN += wNN.MSIE(datasetAR, imputed_wNN_AR)
    MAIE_AR_wNN += wNN.MAIE(datasetAR, imputed_wNN_AR)

    MSIE_BW_wNN += wNN.MSIE(datasetBW, imputed_wNN_BW)
    MAIE_BW_wNN += wNN.MAIE(datasetBW, imputed_wNN_BW)


    MSIE_AR_wNNC += wNNC.MSIE(datasetAR, imputed_wNNC_AR)
    MAIE_AR_wNNC += wNNC.MAIE(datasetAR, imputed_wNNC_AR)

    MSIE_BW_wNNC += wNNC.MSIE(datasetBW, imputed_wNNC_BW)
    MAIE_BW_wNNC += wNNC.MAIE(datasetBW, imputed_wNNC_BW)



print("******************************** AR ********************************")
print("                     kNN                 wNN                    wNNCorrelation")
print(f" MSIE    {MSIE_AR_kNN/samples}      {MSIE_AR_wNN/samples}       {MSIE_AR_wNNC/samples}")
print(f" MAIE    {MAIE_AR_kNN/samples}      {MAIE_AR_wNN/samples}       {MAIE_AR_wNNC/samples}")

print()
print()

print("******************************** BW ********************************")
print("                 kNN                     wNN                    wNNCorrelation")
print(f" MSIE    {MSIE_BW_kNN/samples}      {MSIE_BW_wNN/samples}       {MSIE_BW_wNNC/samples}")
print(f" MAIE    {MAIE_BW_kNN/samples}      {MAIE_BW_wNN/samples}       {MAIE_BW_wNNC/samples}")
