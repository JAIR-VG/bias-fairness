import os

import pandas as pd
from scipy.stats import zscore
from aif360.datasets import StandardDataset



class GliomaDatasetMulti(StandardDataset):
    """Glioma Dataset.
    Class
    0 = 487 (positive)
    1 =  352 (negative)

    Gender
    0 = Male 488
    1 = Female 351

    Race
    0 = 765
    1 = 59
    2 = 14
    3 = 1
    """

    def default_preprocessing(df):
        df['Age_Normalized']=df.filter(like="Age_at_diagnosis").apply(zscore)
        return df

    def __init__(self, label_name='Grade',
                 favorable_classes=[0],
                 protected_attribute_names=['Gender','Race'],
                 privileged_classes=[[0],[0]],
                 instance_weights_name=None,
                 categorical_features=['IDH1', 'TP53', 'ATRX',
                                       'PTEN', 'EGFR', 'CIC', 'MUC16',
                                       'PIK3CA','NF1', 'PIK3R1', 'FUBP1',	
                                       'RB1', 'NOTCH1', 'BCOR', 'CSMD3',
                                       'SMARCA4', 'GRIN2A',	'IDH2', 'FAT4',	'PDGFRA'
                                       ],
                 features_to_keep=[], features_to_drop=['Age_at_diagnosis'],
                 na_values=[], custom_preprocessing=default_preprocessing,
                 metadata=None):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data', 'glioma', 'TCGA_InfoWithGrade.csv')
        
        try:
            df = pd.read_csv(filepath, header=0)
          
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the datasets from:")
            print("\n\thttps://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'glioma'))))
            import sys
            sys.exit(1)


        super(GliomaDatasetMulti, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
