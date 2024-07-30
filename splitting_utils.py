
from collections import OrderedDict
import numpy as np
import pandas as pd


def compute_feature_class(dataset_m,
                          unprivileged_groups, privileged_groups):
    
    df = pd.DataFrame(dataset_m.features,columns=dataset_m.feature_names)
    print(df)
    return df
    #print(dataset_m.features.shape)
    #y=dataset_m.labels
    #x= dataset_m.features
    #print(type(x))
    #print(type(y))
    #print(y.shape)
    #print(y.ndim)