import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd

from aif360.datasets import GermanDataset

np.random.seed(0)

dataset_orig=GermanDataset()

#print(dataset_orig)

print(dataset_orig.feature_names)

