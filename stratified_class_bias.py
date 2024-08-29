import numpy as np
import pandas as pd

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
      
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dorig_train, dorig_test = dorig.split([0.8],shuffle=True)

print(dorig_test.labels)
print(dorig_test.labels.ravel()) 
#convierte en el formato requerido

iris = datasets.load_iris() #Loading the datasetç

X = iris.data[:, [2, 3]]
y = iris.target

print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

print(X_test)
print(y_test)