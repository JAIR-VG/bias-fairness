import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd

from aif360.datasets import AdultDataset

np.random.seed(0)

dataset_orig=AdultDataset()

#print(dataset_orig)

print(dataset_orig.feature_names)