import numpy as np
import pandas as pd

import splitt_utils as su

from aif360.datasets import GermanDataset

from aif360.datasets import AdultDataset

from aif360.datasets import CompasDataset

from aif360.datasets import BankDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21



np.random.seed(1)

#dataset_m = GermanDataset()
#dataset_m = AdultDataset()
#dataset_m = CompasDataset()
#dataset_m = BankDataset()
#dataset_m = MEPSDataset19()
#dataset_m = MEPSDataset20()
dataset_m = MEPSDataset21()
#privileged_groups = [{'sex': 1}]
#unprivileged_groups = [{'sex': 0}]

#privileged_groups = [{'race': 1}]
#unprivileged_groups = [{'race': 0}]
privileged_groups = [{'RACE': 1}]
unprivileged_groups = [{'RACE': 0}]
#privileged_groups = [{'age': 1}]
#unprivileged_groups = [{'age': 0}]

sens_ind = 0




##print(dataset_m.feature_names)

#df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)
#print(df.head())

#Nombre de la columna/atributo protegido
#sens_attr = dataset_m.protected_attribute_names[sens_ind]

#NOmbre de la columna/atributo de clase
#label_attr =dataset_m.label_names[0]

#print(sens_attr)
#print(label_attr)
#print(dataset_m.labels)


#Etiquetas de Clase
#print(dataset_m.favorable_label)
#print(dataset_m.unfavorable_label)

#print(dataset_m.privileged_protected_attributes[sens_ind])
#print(dataset_m.unprivileged_protected_attributes[sens_ind])

#x=df[sens_attr].values.tolist()
#y=dataset_m.labels
#y=np.squeeze(y)
#y=list(y)


#print(type(x))
#print(type(y))

#print(x.count(0))
#print(x.count(1))

#print(y[0])

#print(y.count(dataset_m.favorable_label))
#print(y.count(dataset_m.unfavorable_label))

#print(y[1])
 
#lista = pd.DataFrame({'Protected':x, 'Etiqueta':y})


#new = lista.groupby(['Protected','Etiqueta']).size()

new = su.compute_feature_class(dm=dataset_m,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind)

print(new)

resultado=su.compute_distribution_feature(dataset_m,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

print(resultado)
#print(type(new))

df_train, df_test = dataset_m.split([0.8],shuffle=True)

new = su.compute_feature_class(dm=df_train,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind)


print(new)



new = su.compute_feature_class(dm=df_test,
                               unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               sens_ind=sens_ind)


print(new)


