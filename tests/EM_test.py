import sys
sys.path.append('..')

import math
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from methods.EM import EM
from methods.utils import *

# ************************ VARIABLES *******************************
# REPETITIONS
samples = 1

# GENERATE DATASETS
n = 100
d = 10
#   AR
rho = 0.9
#   BW
rho_w = 0.9
rho_b = 0.1
predictors_per_block = 5

# MISSINGNESS
missingness_percentage = 0.05

# ******************************************************************

model = EM()

for _ in tqdm(range(samples)):
    datasetAR = generate_dataset_AR(n, d, rho)
    datasetBW = generate_dataset_blockwise(n, d, rho_w, rho_b, predictors_per_block)

    datasetAR, datasetAR_missing = generate_missingness_flatten(datasetAR, missingness_percentage)
    datasetBW, datasetBW_missing = generate_missingness_flatten(datasetBW, missingness_percentage)

    _, _, _, imputed_dataset = model.impute(copy.deepcopy(datasetAR_missing), 1)

    print("MSIE AR => ", model.MSIE(datasetAR, imputed_dataset))
    print("MAIE AR => ", model.MAIE(datasetAR, imputed_dataset))

    # total_missing_values = np.count_nonzero(abs(datasetAR - imputed_dataset))
    # print(math.sqrt(np.sum(abs(datasetAR - imputed_dataset) ** 2)/total_missing_values))
    #
    # imputed_dataset = impute_em(datasetAR_missing)
    # total_missing_values = np.count_nonzero(abs(datasetAR - imputed_dataset['X_imputed']))
    # print(math.sqrt(np.sum(abs(datasetAR - imputed_dataset['X_imputed']) ** 2)/total_missing_values))


    priors, mus, covs, imputed_dataset = model.impute(datasetBW_missing, 1)

    print("MSIE BW => ", model.MSIE(datasetBW, imputed_dataset))
    print("MAIE BW => ", model.MAIE(datasetBW, imputed_dataset))
