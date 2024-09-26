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
"""
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
dataset_orig = load_preproc_data_compas(['race'])
"""


sens_ind = 0

#Calculo sobre el conjunto original

new = SU.compute_feature_class(dm=dataset_orig,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind)

print(new)

