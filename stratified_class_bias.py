import numpy as np
import pandas as pd

import splitt_utils as my_su

from sklearn.model_selection import train_test_split
from sklearn import datasets

from aif360.metrics import ClassificationMetric
from aif360.datasets import GermanDataset

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression as LG
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier as RF


dorig = GermanDataset(
    protected_attribute_names=['sex'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[['male']],#,      # age >=25 is considered privileged
    features_to_drop=['personal_status'] # ignore sex-related attributes
)
      

n = dorig.features.shape[0]      
print("Valor de n = ",n)
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
shuffle = False

#print(dorig.instance_names)
#print(dorig.labels.ravel())

skf=StratifiedKFold(n_splits=5)
sens_ind = 0
X=dorig.features
y=dorig.labels.ravel()
for i, (train_index,test_index) in enumerate(skf.split(X,y)):
        print(f"Fold {i}: ")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        type(train_index)
        type(test_index)
        dtra,dtest = my_su.get_folds(dorig,train_index,test_index)
        resultado =my_su.compute_feature_class(dm=dtra,unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups,
                                               sens_ind=sens_ind)
        print("Resultado de entrenamiento ")
        print(resultado)

        #print(dtest.instance_names)


#order = list(np.random.permutation(n) if shuffle else range(n))

#print(dorig.features)

#features = np.array_split(dorig.features[order], 2)

#print(features)

#dorig_train, dorig_test = dorig.split([0.8],shuffle=True)

#print(dorig_test.labels)
#print(dorig_test.labels.ravel()) 
#convierte en el formato requerido

#iris = datasets.load_iris() #Loading the datasetç

#X = iris.data[:, [2, 3]]
#y = iris.target

#print(y)

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

#print(X_test)
#print(y_test)


