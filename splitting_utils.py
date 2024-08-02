
from collections import OrderedDict
import numpy as np
import pandas as pd


def compute_feature_class(dataset_m, unprivileged_groups, privileged_groups,sens_ind):
    
    
    keyp=list((privileged_groups[0]).keys())

    valuep = list((privileged_groups[0]).values())
    
    valueunp = list((unprivileged_groups[0]).values())
    
    sens_attr =keyp[0]
    #print(valuep[0])
    #print(sens_attr)

    valuep = valuep[0]
    valueunp = valueunp[0]
    

    #Nombre de la columna/atributo protegido
    #sens_attr = dataset_m.protected_attribute_names[sens_ind]

    df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)

    x=df[sens_attr].values.tolist()
    y=dataset_m.labels
    y=np.squeeze(y)
    y=list(y)
    
    
    ncountp=x.count(valuep)
    ncountunp =x.count(valueunp)

    print('Number of Unprivileged Values =', ncountunp)
    print('Number of Privileged =',ncountp)

    lista = pd.DataFrame({'Protected':x, 'Etiqueta':y})
    resume = lista.groupby(['Protected','Etiqueta']).size()

    return resume

    #print(df)
    #return df
    #print(dataset_m.features.shape)
    #y=dataset_m.labels
    #x= dataset_m.features
    #print(type(x))
    #print(type(y))
    #print(y.shape)
    #print(y.ndim)

def double_splitting(dataset_m, unprivileged_groups, privileged_groups,sens_ind):

    keyp=list((privileged_groups[0]).keys())

    valuep = list((privileged_groups[0]).values())
    
    valueunp = list((unprivileged_groups[0]).values())
    
    sens_attr =keyp[0]
    #print(valuep[0])
    #print(sens_attr)

    valuep = valuep[0]
    valueunp = valueunp[0]
    
    df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)

    x=df[sens_attr].values.tolist()
    y=dataset_m.labels
    y=np.squeeze(y)
    y=list(y)
    print(x)
    print(y)
