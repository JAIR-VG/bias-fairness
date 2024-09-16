import numpy as np
import pandas as pd

import splitt_utils as su

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets import BankDataset
from aif360.datasets import AdultDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import CompasDataset
from aif360.datasets import GermanDataset
#dataset_orig =BankDataset()

#privileged_groups = [{'race': 1}]
#unprivileged_groups = [{'race': 0}]
#dataset_orig = load_preproc_data_compas(['race'])

# print out some labels, names, etc.
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


#from aif360.datasets import GermanDataset

dataset_m = GermanDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

resultado =su.compute_distribution_feature(dm=dataset_m,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

print(resultado)

sens_ind = 0
resultado = su.compute_feature_class(dm=dataset_m,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups,sens_ind=sens_ind)
print(resultado)
#su.double_split(dm=dataset_m,unprivileged_groups=unprivileged_groups,
#                privileged_groups=privileged_groups,num_or_size_splits=[0.8],
#                shuffle=False,seed=0)
#sens_ind = 0
#new = su.compute_feature_class(dm=dataset_m,
#                               unprivileged_groups=unprivileged_groups,
#                               privileged_groups=privileged_groups,
#                               sens_ind=sens_ind)

#print(new)

#dtrain,dtest=dataset_m.split(num_or_size_splits=[0.8],shuffle=True)
#print(dtrain.labels.ravel())
#new = su.compute_feature_class(dm=dtrain,
#                               unprivileged_groups=unprivileged_groups,
#                               privileged_groups=privileged_groups,
#                               sens_ind=sens_ind)

#print(new)

#print('Tamanyo Particion Propuesta Originl TRA ',len(dtrain.features))
#print('Tamanyo Particion Propuesta Originl Test',len(dtest.features))

#dtrain2,dtest2 =su.double_split(dm=dataset_m,unprivileged_groups=unprivileged_groups,
#                                privileged_groups=privileged_groups,num_or_size_splits=[0.8],
#                                shuffle=True,seed=0)



#new = su.compute_feature_class(dm=dtrain2,
#                               unprivileged_groups=unprivileged_groups,
#                               privileged_groups=privileged_groups,
#                               sens_ind=sens_ind)

#print(new)
#print('Tamanyo mio Train ',len(dtrain2.features))
#print('Tamanyo mio Test ',len(dtest2.features))


#print(len(dtest2.features))
#n=dataset_m.features.shape[0]

#print(n)

#num_or_size_splits =[0.7]

#shuffle = True

#seed =0


#print(dataset_m.features)
#print(dataset_m.labels)
#features = np.array_split(dataset_m.features[order], num_or_size_splits)
#labels = np.array_split(dataset_m.labels[order], num_or_size_splits)
#print(labels)
#sens_ind = 0

#keyp=list((privileged_groups[sens_ind]).keys())

#print(keyp)

#valuep = list((privileged_groups[sens_ind]).values())

#print(valuep)

#valueunp = list((unprivileged_groups[sens_ind]).values())

#print(valueunp)

#Obtiene los valores simples y no como una lista
#sens_attr =keyp[sens_ind]
#valuep = valuep[sens_ind]
#valueunp = valueunp[sens_ind]

#df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)

#print(df)

#x=df[sens_attr].values.tolist()

#print(x)
#y=dataset_m.labels
#y=np.squeeze(y)
#y=list(y)
#print(y)

#ncountp=x.count(valuep)
#ncountunp =x.count(valueunp)

#print('Number of Unprivileged Values =', ncountunp)
#print('Number of Privileged =',ncountp)

#df = pd.DataFrame({'Protected':x, 'Etiqueta':y})
#print(df.index)
#df['index'] = df.index
#print(df)

#lista=df.sort_values(['Protected','Etiqueta'])

#reducedlista = lista.loc[(lista["Protected"] == 0) & (lista["Etiqueta"] ==1)]

#print(reducedlista)