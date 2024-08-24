
from collections import OrderedDict
import numpy as np
import pandas as pd


def compute_distribution_feature (dm, unprivileged_groups, privileged_groups):
    
    df = pd.DataFrame(dm.features,columns=dm.feature_names)
    
    #We get class labels
    y=dm.labels
    y=np.squeeze(y)
    y=list(y)
    
    #Insert class labels
    df.insert(loc=len(df.columns),column='ClassLabel',value=y)
 
    keyp=list((privileged_groups[0]).keys())

    print(keyp)

    valuep = list((privileged_groups[0]).values())
    
    print(valuep)
    valueunp = list((unprivileged_groups[0]).values())
    print(valueunp)
    
    
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


    idx_priv_fav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==fav_label)].tolist()
    idx_priv_unfav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==unfav_label)].tolist()
    idx_unpriv_fav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==fav_label)].tolist()
    idx_unpriv_unfav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==unfav_label)].tolist()

    texto1= "Protected Attribute("+sens_attr+"), Unpriviliged("+str(unpriv_value)+") + Unfavorable Class("+str(unfav_label)+")"
    texto2= "Protected Attribute("+sens_attr+"), Unpriviliged("+str(unpriv_value)+") + Favorable Class("+str(fav_label)+")"
    texto3= "Protected Attribute("+sens_attr+"), Priviliged("+str(priv_value)+") + Unfavorable Class("+str(unfav_label)+")"
    texto4= "Protected Attribute("+sens_attr+"), Priviliged("+str(priv_value)+") + Favorable Class("+str(fav_label)+")"

    resume={
        texto1:len(idx_unpriv_unfav),
        texto2:len(idx_unpriv_fav),
        texto3:len(idx_priv_unfav),
        texto4:len(idx_unpriv_unfav)
    }

    return resume


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
    
    n = dm.features.shape[0]

    if isinstance(num_or_size_splits, list):
        num_folds = len(num_or_size_splits) + 1
        #if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
        #    num_or_size_splits = [int(x * n) for x in num_or_size_splits]
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
    n_priv_fav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==fav_label)].tolist()
    n_priv_unfav = df.index[(df[sens_attr] == priv_value) & (df['ClassLabel']==unfav_label)].tolist()
    n_unpriv_fav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==fav_label)].tolist()
    n_unpriv_unfav = df.index[(df[sens_attr] == unpriv_value) & (df['ClassLabel']==unfav_label)].tolist()

    
    order_priv_fav = list(np.random.permutation(n_priv_fav) if shuffle else n_priv_fav)
    order_priv_unfav = list(np.random.permutation(n_priv_unfav) if shuffle else n_priv_unfav)
    order_unpriv_fav = list(np.random.permutation(n_unpriv_fav) if shuffle else n_unpriv_fav)
    order_unpriv_unfav = list(np.random.permutation(n_unpriv_unfav) if shuffle else n_unpriv_unfav)

    folds = [dm.copy() for _ in range(num_folds)]

    if isinstance(num_or_size_splits, list):
        if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
            num_or_size_splits_priv_fav = [int(x * len(n_priv_fav)) for x in num_or_size_splits]
            num_or_size_splits_priv_unfav = [int(x * len(n_priv_unfav)) for x in num_or_size_splits]
            num_or_size_splits_unpriv_fav = [int(x * len(n_unpriv_fav)) for x in num_or_size_splits]
            num_or_size_splits_unpriv_unfav = [int(x * len(n_unpriv_unfav)) for x in num_or_size_splits]
            num_or_size_splits = [int(x * n) for x in num_or_size_splits]
    else:
        num_or_size_splits_priv_fav = num_or_size_splits
        num_or_size_splits_priv_unfav = num_or_size_splits
        num_or_size_splits_unpriv_fav = num_or_size_splits
        num_or_size_splits_unpriv_unfav = num_or_size_splits
        num_or_size_splits = num_or_size_splits


    #print(len(n_priv_fav))
    #print(len(n_priv_unfav))
    #print(len(n_unpriv_fav))
    #print(len(n_unpriv_unfav))

    #print(order_priv_fav)

    #print(num_or_size_splits)
    #print(dm.features[order_priv_fav])
    feat_priv_fav = np.array_split(dm.features[order_priv_fav],
                                   num_or_size_splits_priv_fav)
    lab_priv_fav = np.array_split(dm.labels[order_priv_fav],
                                  num_or_size_splits_priv_fav)
    scor_priv_fav = np.array_split(dm.scores[order_priv_fav],
                                   num_or_size_splits_priv_fav)
    prot_att_priv_fav = np.array_split(dm.protected_attributes[order_priv_fav],
                                       num_or_size_splits_priv_fav)
    inst_w_priv_fav = np.array_split(dm.instance_weights[order_priv_fav],
                                     num_or_size_splits_priv_fav)
    inst_nam_priv_fav = np.array_split(np.array(dm.instance_names)[order_priv_fav],
                                       num_or_size_splits_priv_fav)
    
    feat_priv_unfav = np.array_split(dm.features[order_priv_unfav],
                                     num_or_size_splits_priv_unfav)
    lab_priv_unfav = np.array_split(dm.labels[order_priv_unfav],
                                    num_or_size_splits_priv_unfav)
    scor_priv_unfav = np.array_split(dm.scores[order_priv_unfav],
                                     num_or_size_splits_priv_unfav)
    prot_att_priv_unfav = np.array_split(dm.protected_attributes[order_priv_unfav],
                                     num_or_size_splits_priv_unfav)
    inst_w_priv_unfav = np.array_split(dm.instance_weights[order_priv_unfav],
                                       num_or_size_splits_priv_unfav)
    inst_nam_priv_unfav = np.array_split(np.array(dm.instance_names)[order_priv_unfav],
                                         num_or_size_splits_priv_unfav)


    feat_unpriv_fav = np.array_split(dm.features[order_unpriv_fav],
                                   num_or_size_splits_unpriv_fav)
    lab_unpriv_fav = np.array_split(dm.labels[order_unpriv_fav],
                                  num_or_size_splits_unpriv_fav)
    scor_unpriv_fav = np.array_split(dm.scores[order_unpriv_fav],
                                   num_or_size_splits_unpriv_fav)
    prot_att_unpriv_fav = np.array_split(dm.protected_attributes[order_unpriv_fav],
                                       num_or_size_splits_unpriv_fav)
    inst_w_unpriv_fav = np.array_split(dm.instance_weights[order_unpriv_fav],
                                     num_or_size_splits_unpriv_fav)
    inst_nam_unpriv_fav = np.array_split(np.array(dm.instance_names)[order_unpriv_fav],
                                       num_or_size_splits_unpriv_fav)
    

    feat_unpriv_unfav = np.array_split(dm.features[order_unpriv_unfav],
                                       num_or_size_splits_unpriv_unfav)
    lab_unpriv_unfav = np.array_split(dm.labels[order_unpriv_unfav],
                                      num_or_size_splits_unpriv_unfav)
    scor_unpriv_unfav = np.array_split(dm.scores[order_unpriv_unfav],
                                       num_or_size_splits_unpriv_unfav)
    prot_att_unpriv_unfav = np.array_split(dm.protected_attributes[order_unpriv_unfav],
                                     num_or_size_splits_unpriv_unfav)
    inst_w_unpriv_unfav = np.array_split(dm.instance_weights[order_unpriv_unfav],
                                       num_or_size_splits_unpriv_unfav)
    inst_nam_unpriv_unfav = np.array_split(np.array(dm.instance_names)[order_unpriv_unfav],
                                         num_or_size_splits_unpriv_unfav)

    print('Size feat_priv_fav ',len(feat_priv_fav[0]))
    #print('Size feat_priv_fav ',len(feat_priv_fav[1]))
    print('Size feat_priv_unfav ',len(feat_priv_unfav[0]))
    print('Size feat_unpriv_fav ',len(feat_unpriv_fav[0]))
    print('Size feat_unpriv_unfav ',len(feat_unpriv_unfav[0]))

    #print(feat_priv_fav)
    features = feat_priv_fav
    labels = lab_priv_fav
    scores = scor_priv_fav
    protected_attributes = prot_att_priv_fav
    instance_weights = inst_w_priv_fav
    instance_names = inst_nam_priv_fav
    
   # print('Num folds',num_folds)
    for i in range(num_folds):
        features[i] = np.concatenate((feat_priv_fav[i],feat_priv_unfav[i],
                                feat_unpriv_fav[i],feat_unpriv_unfav[i]),
                                axis=0)
    #    print(features[i])
        labels[i]=np.concatenate((lab_priv_fav[i],lab_priv_unfav[i],
                            lab_unpriv_fav[i],lab_unpriv_unfav[i]),
                            axis=0)
        scores[i]=np.concatenate((scor_priv_fav[i],scor_priv_unfav[i],
                            scor_unpriv_fav[i],scor_unpriv_unfav[i]),
                            axis=0)
        protected_attributes[i]=np.concatenate((prot_att_priv_fav[i],prot_att_priv_unfav[i],
                                          prot_att_unpriv_fav[i],prot_att_unpriv_unfav[i]),
                                          axis=0)
        instance_weights[i]=np.concatenate((inst_w_priv_fav[i],inst_w_priv_unfav[i],
                                      inst_w_unpriv_fav[i],inst_w_unpriv_unfav[i]),
                                      axis=0)
        instance_names[i] = np.concatenate((inst_nam_priv_fav[i],inst_nam_priv_unfav[i],
                                      inst_nam_unpriv_fav[i],inst_nam_unpriv_unfav[i]),
                                      axis=0)
    
 #   print('Features',len(features))
        
    for fold, feats, labs, scors, prot_attrs, inst_wgts, inst_name in zip(
        folds, features, labels, scores, protected_attributes, instance_weights,
        instance_names):
        fold.features = feats
        fold.labels = labs
        fold.scores = scors
        fold.protected_attributes = prot_attrs
        fold.instance_weights = inst_wgts
        fold.instance_names = list(map(str, inst_name))
        fold.metadata = fold.metadata.copy()
        fold.metadata.update({
            'transformer': '{}.split'.format(type(dm).__name__),
            'params': {'num_or_size_splits': num_or_size_splits,
                       'shuffle': shuffle},
                       'previous': [dm]
                       })

    return folds
#Si num_or_size_splits es un número entero que indica folds, crea n-folds. Si es 5 creará 5 conjuntos
#Si num_or_size_splits es un número entero que indica el número de muestras a elegir, entonces formará un conjunto con que tiene ese número
#Por ejemplo [800] que representa el 80% selecciona [800 muestras] y el resto [200]
#    features = np.array_split(dm.features[order], num_or_size_splits)
