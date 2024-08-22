
from collections import OrderedDict
import numpy as np
import pandas as pd


# Este metodo calcula para un atributo, para cada valor, cuantas muestras hay por clase.
def compute_feature_class(dm, unprivileged_groups, privileged_groups,sens_ind):
    
    
    keyp=list((privileged_groups[sens_ind]).keys())

    valuep = list((privileged_groups[sens_ind]).values())
    
    valueunp = list((unprivileged_groups[sens_ind]).values())
    
    sens_attr =keyp[sens_ind]

    valuep = valuep[sens_ind]
    valueunp = valueunp[sens_ind]
    

    #Nombre de la columna/atributo protegido
    #sens_attr = dm.protected_attribute_names[sens_ind]

    df = pd.DataFrame(dm.features,columns=dm.feature_names)

    x=df[sens_attr].values.tolist()
    y=dm.labels
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
    #print(dm.features.shape)
    #y=dm.labels
    #x= dm.features
    #print(type(x))
    #print(type(y))
    #print(y.shape)
    #print(y.ndim)

def double_split(dm, unprivileged_groups, privileged_groups,num_or_size_splits, shuffle=False,seed =None):

    if seed is not None:
        np.random.seed(seed)
    
   # n = dm.features.shape[0]



    if isinstance(num_or_size_splits, list):
        num_folds = len(num_or_size_splits) + 1
#        if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
 #           num_or_size_splits = [int(x * n) for x in num_or_size_splits]
    else:
        num_folds = num_or_size_splits
        
    df = pd.DataFrame(dm.features,columns=dm.feature_names)

    y=dm.labels
    y=np.squeeze(y)
    y=list(y)
    
    df.insert(loc=len(df.columns),column='ClassLabel',value=y)

 
    keyp=list((privileged_groups[0]).keys())

    valuep = list((privileged_groups[0]).values())
    
    valueunp = list((unprivileged_groups[0]).values())
    
    #Column Name of the protected attribute
    sens_attr =keyp[0]
    #Value of privileged attribute
    priv_value = valuep[0]
    #Value of unpriviliged attribute
    unpriv_value = valueunp[0]

    #Class Label for favorable class
    fav_label = dm.favorable_label
    #Class Label for unfavorable class
    unfav_label = dm.unfavorable_label

  #  print(sens_attr)
  #  print(priv_value)
  #  print(unpriv_value)
  #  print(fav_label)
  #  print(unfav_label)
   # print(df[sens_attr])

    #Indices considerando valores de atributos 
    idx_priv_fav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==fav_label)].tolist()
    idx_priv_unfav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==unfav_label)].tolist()
    idx_unpriv_fav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==fav_label)].tolist()
    idx_unpriv_unfav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==unfav_label)].tolist()

    if shuffle:
        idx_priv_fav = list(np.random.permutation(idx_priv_fav))
        idx_priv_unfav = list(np.random.permutation(idx_priv_unfav))
        idx_unpriv_fav = list(np.random.permutation(idx_unpriv_fav))
        idx_unpriv_unfav = list(np.random.permutation(idx_unpriv_unfav))


    print(len(idx_priv_fav))
    print(len(idx_priv_unfav))
    print(len(idx_unpriv_fav))
    print(len(idx_unpriv_unfav))

    #print([int(x * len(idx_priv_fav)) for x in num_or_size_splits])


    if isinstance(num_or_size_splits, list):
        #Operations for priviliged and favorable
        n_o_s_s = [int(x * len(idx_priv_fav)) for x in num_or_size_splits]
        print(n_o_s_s)
        feat_priv_fav = np.array_split(dm.features[idx_priv_fav],
                                       n_o_s_s)
        lab_priv_fav = np.array_split(dm.labels[idx_priv_fav],
                                      n_o_s_s)
        scor_priv_fav = np.array_split(dm.scores[idx_priv_fav],
                                       n_o_s_s)
        prot_att_priv_fav = np.array_split(dm.protected_attributes[idx_priv_fav],
                                           n_o_s_s)
        inst_w_priv_fav = np.array_split(dm.instance_weights[idx_priv_fav],
                                         n_o_s_s)
        inst_nam_priv_fav = np.array_split(np.array(dm.instance_names)[idx_priv_fav],
                                           n_o_s_s)

        
        #Operations for priviliged and unfavorable
        n_o_s_s = [int(x * len(idx_priv_unfav)) for x in num_or_size_splits]
        print(n_o_s_s)
        feat_priv_unfav = np.array_split(dm.features[idx_priv_unfav],
                                         n_o_s_s)
        lab_priv_unfav = np.array_split(dm.labels[idx_priv_unfav],
                                        n_o_s_s)
        scor_priv_unfav = np.array_split(dm.scores[idx_priv_unfav],
                                         n_o_s_s)
        prot_att_priv_unfav = np.array_split(dm.protected_attributes[idx_priv_unfav],
                                             n_o_s_s)
        inst_w_priv_unfav = np.array_split(dm.instance_weights[idx_priv_unfav],
                                           n_o_s_s)
        inst_nam_priv_unfav = np.array_split(np.array(dm.instance_names)[idx_priv_unfav],
                                             n_o_s_s)

         #Operations for unpriviliged and favorable
        n_o_s_s = [int(x * len(idx_unpriv_fav)) for x in num_or_size_splits]
        print(n_o_s_s)
        feat_unpriv_fav = np.array_split(dm.features[idx_unpriv_fav],
                                         n_o_s_s)
        lab_unpriv_fav = np.array_split(dm.labels[idx_unpriv_fav],
                                        n_o_s_s)
        scor_unpriv_fav = np.array_split(dm.scores[idx_unpriv_fav],
                                         n_o_s_s)
        prot_att_unpriv_fav = np.array_split(dm.protected_attributes[idx_unpriv_fav],
                                             n_o_s_s)
        inst_w_unpriv_fav = np.array_split(dm.instance_weights[idx_unpriv_fav],
                                           n_o_s_s)
        inst_nam_unpriv_fav = np.array_split(np.array(dm.instance_names)[idx_unpriv_fav],
                                             n_o_s_s)

         #Operations for unpriviliged and unfavorable
        n_o_s_s = [int(x * len(idx_unpriv_unfav)) for x in num_or_size_splits]
        print(n_o_s_s)
        feat_unpriv_unfav = np.array_split(dm.features[idx_unpriv_unfav],
                                           n_o_s_s)
        lab_unpriv_unfav = np.array_split(dm.labels[idx_unpriv_unfav],
                                          n_o_s_s)
        scor_unpriv_unfav = np.array_split(dm.scores[idx_unpriv_unfav],
                                           n_o_s_s)
        prot_att_unpriv_unfav = np.array_split(dm.protected_attributes[idx_unpriv_unfav],
                                               n_o_s_s)
        inst_w_unpriv_unfav = np.array_split(dm.instance_weights[idx_unpriv_unfav],
                                             n_o_s_s)
        inst_nam_unpriv_unfav = np.array_split(np.array(dm.instance_names)[idx_unpriv_unfav],
                                               n_o_s_s)

        feat_tra=np.append(feat_priv_fav[0],feat_priv_unfav[0],
                           feat_unpriv_fav[0],feat_unpriv_unfav,
                           axis=0)
        print(feat_tra.shape)
        
    #print(type(feat_priv_fav))
    #print(len(feat_priv_fav))
    #print(feat_priv_fav[0].shape)
    #print(feat_priv_fav[1].shape)
    #np_newarray=np.append(feat_priv_fav[0],feat_priv_fav[1],axis=0)
    #print(np_newarray.shape)
    #print(np_newarray)
        
        #labels = np.array_split(self.labels[order], num_or_size_splits)

    #print(idx_priv_unfav)
    #print(idx_unpriv_fav)
    #print(idx_unpriv_unfav)
    #print(dm.labels)
    #print(np.unique(np.array(dm.labels)))


    
    #print(df)
    #Variable importante es num_or_size splits

    
    
    #order = list(np.random.permutation(n) if shuffle else range(n))
    
    
    # Crea n copias del dataaset
    #folds = [dm.copy() for _ in range(num_folds)]



    #print(folds)
    #print(dm.favorable_label, dm.unfavorable_label)
    #print(dm.privileged_protected_attributes,
    #      dm.unprivileged_protected_attributes)
    #print(dm.protected_attributes)
    #print(dm.protected_attribute_names)
    #print(dm.feature_names)
    #print(dm.features)

#Si num_or_size_splits es un número entero que indica folds, crea n-folds. Si es 5 creará 5 conjuntos
#Si num_or_size_splits es un número entero que indica el número de muestras a elegir, entonces formará un conjunto con que tiene ese número
#Por ejemplo [800] que representa el 80% selecciona [800 muestras] y el resto [200]
#    features = np.array_split(dm.features[order], num_or_size_splits)
