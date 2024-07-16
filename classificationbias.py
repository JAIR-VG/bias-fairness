from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


df = pd.read_csv('Germancoded.csv')

column_names = list(df.columns)
column_names.append('LabelPred')
#print(column_names)

#headerString= ",".join(column_names)

#print(df)

#print(type(df))

#print(df.Label)
xpd =df.loc[:,df.columns!= 'Label']
X=xpd.to_numpy()
xpd=df.iloc[:,-1]
y=xpd.to_numpy()
#print(Y)
#print(len(X))
#print(len(X[0]))
#print(X[0])
T=df.to_numpy()

#Classification with StratifiedKFold
skf=StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
skf.get_n_splits(X,y)

clf = RandomForestClassifier(random_state=0)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    X_train_fold, y_train_fold = X[train_index],y[train_index]
    X_test_fold, y_test_fold = X[test_index],y[test_index]

    Test = T[test_index]

    clf.fit(X_train_fold,y_train_fold)
    
    ypred = clf.predict(X_test_fold)

    #print(len(Test))
    #print(len(ypred))
    #print(len(Test[0]))
    #print(ypred[1])
    
    result = np.c_[Test,ypred]
    #print(ypred)
    

   # print(result)
    
    DF = pd.DataFrame(result,columns=column_names)
    #print(DF)
    fichero ='ResultsGerman-'+str(i)+'-RF.csv'
    DF.to_csv(fichero,index=False)

   # print(np.array(column_names))

  #  np.savetxt("foo.csv",result,header =headerString,delimiter=",")


    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")

