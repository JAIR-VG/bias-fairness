import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd
import airlinedataset as aire
#from aif360.datasets import AdultDataset


np.random.seed(0)

dataset_orig=aire.AirlineDataset()

#print(dataset_orig)


print("#### Training Dataset shape")
print(dataset_orig.features.shape,type(dataset_orig.features))

print("#### Favorable and unfavorable labels")
print(dataset_orig.favorable_label, dataset_orig.unfavorable_label)

print("#### Protected attribute names")
print(dataset_orig.protected_attribute_names)

print("#### Privileged and unprivileged protected attribute values")
print(dataset_orig.privileged_protected_attributes, 
      dataset_orig.unprivileged_protected_attributes)



print("#### Dataset feature names")
print(dataset_orig.feature_names)
print()