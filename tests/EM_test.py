import sys
sys.path.append('..')

import math
import copy
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from methods.EM import EM
from methods.EMparallel import EMP
from methods.utils import *

# ************************ VARIABLES *******************************
# REPETITIONS
samples = 2

# GENERATE DATASETS
n = 500
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
modelEMP = EMP()
import time
for _ in tqdm(range(samples)):
    datasetAR = generate_dataset_AR(n, d, rho)

    datasetAR, datasetAR_missing = generate_missingness_flatten(datasetAR, missingness_percentage)

    start = time.time()
    priorsEMP, musEMP, covsEMP, imputed_datasetEMP = modelEMP.impute(copy.deepcopy(datasetAR_missing), 5)
    print(time.time() - start)
    print()

    # start = time.time()
    # priorsEM, musEM, covsEM, imputed_datasetEM = model.impute(copy.deepcopy(datasetAR_missing), 5)
    # print(time.time() - start)
    # print()
    # print()
