import numpy as np
import pandas as pd

import splitt_utils as su

from aif360.datasets import GermanDataset

dataset_m = GermanDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

su.double_splitting(dataset_m=dataset_m,num_or_size_splits=[0.8],
                    unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups,
                    sens_ind=0,shuffle=True,seed=0)

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