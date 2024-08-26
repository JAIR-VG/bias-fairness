import numpy as np
import pandas as pd

import splitt_utils as SU

from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas

from aif360.datasets import GermanDataset
from aif360.datasets import AdultDataset
from aif360.datasets import BankDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import CompasDataset

from sklearn.linear_model import LogisticRegression as LG
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier as RF

"""
dataset_orig = GermanDataset(
    protected_attribute_names=['age'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
)
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]
"""

"""
dataset_orig = GermanDataset(
    protected_attribute_names=['sex'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[['male']],#,      # age >=25 is considered privileged
    features_to_drop=['personal_status'] # ignore sex-related attributes
)
      
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
"""
"""
dataset_orig = BankDataset()
      
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

"""
"""
dataset_orig = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
"""
"""
dataset_orig = AdultDataset(protected_attribute_names=['race'],
                            privileged_classes=[['White']],
                            features_to_keep=['age', 'education-num'])

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
"""
"""
dataset_orig = MEPSDataset19(protected_attribute_names=['RACE'],
                            privileged_classes=[['White']])



privileged_groups = [{'RACE': 1}]
unprivileged_groups = [{'RACE': 0}]
"""
"""
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
dataset_orig = load_preproc_data_compas(['sex'])
"""
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
dataset_orig = load_preproc_data_compas(['race'])

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed=0)
#print("#### Training Dataset shape")
#print(dataset_orig.features.shape,type(dataset_orig.features))

#print("#### Favorable and unfavorable labels")
#print(dataset_orig.favorable_label, dataset_orig.unfavorable_label)

#print("#### Protected attribute names")
#print(dataset_orig.protected_attribute_names)

#print("#### Privileged and unprivileged protected attribute values")
#print(dataset_orig.privileged_protected_attributes, 
#      dataset_orig.unprivileged_protected_attributes)

#print("#### Dataset feature names")
#print(dataset_orig.feature_names)
#print()

sens_ind = 0


#resultado=SU.compute_distribution_feature(dm=dataset_orig,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

new = SU.compute_feature_class(dm=dataset_orig,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind)

print(new)

print(SU.compute_feature_class(dm=dataset_orig_train,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind))

print(SU.compute_feature_class(dm=dataset_orig_test,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind))


dataset_orig_train2,dataset_orig_test2 =SU.double_split(dm=dataset_orig,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups,
                                                        num_or_size_splits=[0.8],
                                                        shuffle=True,seed=0)

print(SU.compute_feature_class(dm=dataset_orig_train2,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind))

print(SU.compute_feature_class(dm=dataset_orig_test2,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind))


#df = pd.DataFrame(dataset_orig.features,columns=dataset_orig.feature_names)

#print(df[['age']].to_string(index=False))

#print(resultado)
