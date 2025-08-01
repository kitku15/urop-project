# generate_params.py
from itertools import product
import numpy as np

F1_LVThresh = np.arange(400000, 450000, 1000)
F2_sigma = np.arange(10, 12, 0.5)
F2_binaryThresh = np.round(np.arange(0.05, 0.07, 0.005), 4)
F2_circThresh = np.round(np.arange(0.7, 0.91, 0.05), 3)

param_combos = list(product(F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh))

with open("params_list.txt", "w") as f:
    for p in param_combos:
        f.write(" ".join(map(str, p)) + "\n")
