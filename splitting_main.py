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

#Nombre de la columna/atributo protegido
sens_attr = dataset_m.protected_attribute_names[sens_ind]

#NOmbre de la columna/atributo de clase
label_attr =dataset_m.label_names[0]

print(sens_attr)
print(label_attr)
#print(dataset_m.labels)


#Etiquetas de Clase
print(dataset_m.favorable_label)
print(dataset_m.unfavorable_label)

print(dataset_m.privileged_protected_attributes[sens_ind])
print(dataset_m.unprivileged_protected_attributes[sens_ind])

x=df[sens_attr].values.tolist()
y=dataset_m.labels
y=np.squeeze(y)
y=list(y)
#print(y)
#print(x)

#print(df_two)

print(type(x))
print(type(y))

print(x.count(0))
print(x.count(1))

#print(y[0])

print(y.count(dataset_m.favorable_label))
print(y.count(dataset_m.unfavorable_label))

print(y[1])
 
lista = pd.DataFrame({'Protected':x, 'Etiqueta':y})

print(lista)
# display result  
#print(y)

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