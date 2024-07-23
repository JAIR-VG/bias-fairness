import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd
from scipy.stats import zscore

from aif360.datasets import GermanDataset

import gliomadataset as Glioma

dataset_orig = GermanDataset(
    protected_attribute_names=['sex'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[['male']],#,      # age >=25 is considered privileged
    features_to_drop=['personal_status'] # ignore sex-related attributes
)
      
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]


#print(dataset_orig)
#import airlinedataset as aire
#from aif360.datasets import AdultDataset
#import gliomadataset as Glioma


#np.random.seed(0)

dataset_orig=Glioma.GliomaDataset()

#print(dataset_orig)

#print("#### Training Dataset shape")
print(dataset_orig.features.shape,type(dataset_orig.features))

print("#### Favorable and unfavorable labels")
print(dataset_orig.favorable_label, dataset_orig.unfavorable_label)

print(dataset_orig.scores)

#print("#### Protected attribute names")
#print(dataset_orig.protected_attribute_names)

#print("#### Privileged and unprivileged protected attribute values")
#print(dataset_orig.privileged_protected_attributes, 
#      dataset_orig.unprivileged_protected_attributes)

#print("#### Dataset feature names")
#print(dataset_orig.feature_names)
#print()

#df = pd.read_csv('data/glioma/TCGA_InfoWithGrade.csv', header=0)
#print (df.shape)
#print(df.head)
#df['Normalizado']=df.filter(like="Age_at_diagnosis").apply(zscore)
#print(df.head)