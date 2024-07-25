import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric


# Explainers
from aif360.explainers import MetricTextExplainer

# Classifiers
from sklearn.ensemble import RandomForestClassifier

#from aif360.datasets import GermanDataset
import gliomadataset as GLC

np.random.seed(2)

dorig = GLC.GliomaDataset()
#dataset_orig = GermanDataset()

sens_ind = 0

print(dorig.protected_attribute_names)
print(dorig.label_names)
print(dorig.feature_names)

sens_attr = dorig.protected_attribute_names[sens_ind]

print(sens_attr)

unprivileged_groups = [{sens_attr: v} for v in
                       dorig.unprivileged_protected_attributes[sens_ind]]

privileged_groups = [{sens_attr: v} for v in
                     dorig.privileged_protected_attributes[sens_ind]]

print(unprivileged_groups)
print(privileged_groups)


#print(dorig.features)
#print(dorig.labels.ravel())
dorig_train, dorig_test = dorig.split([0.8],shuffle=True)

rf_glioma = RandomForestClassifier(random_state=0)

rf_glioma.fit(dorig_train.features,dorig_train.labels.ravel())

y_pred = rf_glioma.predict(dorig_test.features)

#print(y_pred)

dpred = dorig_test.copy(deepcopy=True)

preds2 = np.expand_dims(y_pred,axis=1)
#print(preds2)

dpred.labels=np.copy(preds2)

metric_bias = BinaryLabelDatasetMetric(dorig_train,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

print('Base rate',metric_bias.base_rate())
print('Consistency',metric_bias.consistency())
print('Disparate Impact',metric_bias.disparate_impact())
print('Mean difference', metric_bias.mean_difference())
print('smoothed empirical difference',metric_bias.smoothed_empirical_differential_fairness())

classified_metric_pred = ClassificationMetric(dorig_test,
                            dpred, 
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)

print('Accuracy =',classified_metric_pred.accuracy())
print('Balanced Accuracy', 0.5*(classified_metric_pred.true_positive_rate()+
                            classified_metric_pred.true_negative_rate()))
print('TPR Privileged ',classified_metric_pred.recall(privileged=True))
print('TPR UnPrivileged ',classified_metric_pred.recall(privileged=False))
print('TNR Privileged ',classified_metric_pred.specificity(privileged=True))
print('TNR UnPrivileged ',classified_metric_pred.specificity(privileged=False))
print('Average abs odds difference', classified_metric_pred.average_abs_odds_difference())
print('Average odds difference', classified_metric_pred.average_odds_difference())
print('Equal oportunity difference', classified_metric_pred.equal_opportunity_difference())
print('True Positive Rate Difference',classified_metric_pred.true_positive_rate_difference())
print('True Negative Rate Difference',classified_metric_pred.specificity(privileged=False)-classified_metric_pred.specificity(privileged=True))
#metric_orig=BinaryLabelDatasetMetric(dataset_orig,unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

#print(metric_orig.mean_difference())
#print(metric_orig.base_rate())
#print(metric_orig.consistency(n_neighbors=5))
#print(metric_orig.smoothed_empirical_differential_fairness())



#explainer_orig = MetricTextExplainer(metric_orig)

#print(explainer_orig.disparate_impact())


#privileged_groups = [{'age': 1}]
#unprivileged_groups = [{'age': 0}]



#print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric.mean_difference())
#print(metric.consistency(n_neighbors=3))
#print(metric.disparate_impact())


#dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


