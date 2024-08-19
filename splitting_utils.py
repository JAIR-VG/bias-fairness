
from collections import OrderedDict
import numpy as np
import pandas as pd


# Este metodo calcula para un atributo, para cada valor, cuantas muestras hay por clase.
def compute_feature_class(dataset_m, unprivileged_groups, privileged_groups,sens_ind):
    
    
    keyp=list((privileged_groups[sens_ind]).keys())

    valuep = list((privileged_groups[sens_ind]).values())
    
    valueunp = list((unprivileged_groups[sens_ind]).values())
    
    sens_attr =keyp[sens_ind]
    #print(valuep[0])
    #print(sens_attr)

    valuep = valuep[sens_ind]
    valueunp = valueunp[sens_ind]
    

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

def double_splitting(dataset_m, num_or_size_splits, unprivileged_groups, privileged_groups,sens_ind, shuffle=False,seed =None):

    if seed is not None:
        np.random.seed(seed)
    
    n = dataset_m.features.shape[0]
    print(n)
    if isinstance(num_or_size_splits, list):
        num_folds = len(num_or_size_splits) + 1
        if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
            num_or_size_splits = [int(x * n) for x in num_or_size_splits]
    else:
        num_folds = num_or_size_splits

    print(num_folds)
    order = list(np.random.permutation(n) if shuffle else range(n))
    print(order)
    folds = [dataset_m.copy() for _ in range(num_folds)]
    print(folds)

    print(dataset_m.protected_attributes)
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
#    print(x)
 #   print(y)
