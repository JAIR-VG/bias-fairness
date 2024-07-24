import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

from aif360.datasets import GermanDataset
#import airlinedataset as Air


#dataset_orig = Air.AirlineDataset()
dataset_orig = GermanDataset()

sens_ind = 1

print(dataset_orig.protected_attribute_names)

sens_attr = dataset_orig.protected_attribute_names[sens_ind]

print(sens_attr)

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_orig.unprivileged_protected_attributes[sens_ind]]

privileged_groups = [{sens_attr: v} for v in
                     dataset_orig.privileged_protected_attributes[sens_ind]]

print(unprivileged_groups)
print(privileged_groups)

metric_orig=BinaryLabelDatasetMetric(dataset_orig,unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

print(metric_orig.mean_difference())

#explainer_orig = MetricTextExplainer(metric_orig)

#print(explainer_orig.disparate_impact())


#privileged_groups = [{'age': 1}]
#unprivileged_groups = [{'age': 0}]



#print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric.mean_difference())
#print(metric.consistency(n_neighbors=3))
#print(metric.disparate_impact())


#dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


