import numpy as np
import pandas as pd

import splitting_utils as su

from aif360.datasets import GermanDataset



dataset_m = GermanDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

sens_ind = 0

##print(dataset_m.feature_names)

df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)
#print(df.head())

sens_attr = dataset_m.protected_attribute_names[sens_ind]
label_attr =dataset_m.label_names[0]
print(sens_attr)




x=df[sens_attr].values.tolist()
y=dataset_m.labels

print(type(x))
print(type(y))

#print(x.count(0))
#print(x.count(1))

#su.compute_feature_class(data_full,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

#print(data_full.features.shape)

#print(data_full.labels)

#print(data_full.features)
#y=data_full.labels
#x= data_full.features

#print(type(x))
#print(type(y))
#print(y.shape)
#print(y.ndim)