import sys
sys.path.insert(1, "../") 
import numpy as np
import pandas as pd

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

np.random.seed(0)

# Classifiers
from sklearn.ensemble import RandomForestClassifier

#from aif360.datasets import GermanDataset
import gliomadatasetmulti as GLCMulti

np.random.seed(2)

df_orig = GLCMulti.GliomaDatasetMulti()
#dataset_orig = GermanDataset()

#sens_ind = 0

print(df_orig.protected_attribute_names)


#print(sens_attr)


#unprivileged_groups = [{'Race': 1},{'Race': 2},{'Race': 3}]
unprivileged_groups = [{'Race': 2}]

privileged_groups = [{'Race': 0}]

print(unprivileged_groups)
print(privileged_groups)

df_train, df_test = df_orig.split([0.7],shuffle=True)

print('Training dataset shape',df_train.features.shape)
print('Test dataset shape',df_test.features.shape)

rf_glioma = RandomForestClassifier(random_state=0)

rf_glioma.fit(df_train.features,df_train.labels.ravel())

y_pred = rf_glioma.predict(df_test.features)

dpred = df_test.copy(deepcopy=True)

preds2 = np.expand_dims(y_pred,axis=1)

dpred.labels=np.copy(preds2)

metric_bias = BinaryLabelDatasetMetric(df_train,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

print('Base rate',metric_bias.base_rate())
print('Consistency',metric_bias.consistency())
print('Disparate Impact',metric_bias.disparate_impact())
print('Mean difference', metric_bias.mean_difference())
print('smoothed empirical difference',metric_bias.smoothed_empirical_differential_fairness())


classified_metric_pred = ClassificationMetric(df_test,
                            dpred, 
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)


print('Confusion matrix ',classified_metric_pred.binary_confusion_matrix(privileged=False))

print('Num positives', classified_metric_pred.num_positives(privileged=False))
print('Num negatives', classified_metric_pred.num_negatives(privileged=False))
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