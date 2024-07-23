import os

import pandas as pd
from scipy.stats import zscore
from aif360.datasets import StandardDataset


class GliomaDataset(StandardDataset):
    """Glioma Dataset.
    """

    def default_preprocessing(df):
        df['Age_Normalized']=df.filter(like="Age_at_diagnosis").apply(zscore)
        return df

    def __init__(self, label_name='Grade',
                 favorable_classes=[0],
                 protected_attribute_names=['Gender'],
                 privileged_classes=[[0]],
                 instance_weights_name=None,
                 categorical_features=['Race', 'IDH1', 'TP53', 'ATRX',
                                       'PTEN', 'EGFR', 'CIC', 'MUC16',
                                       'PIK3CA','NF1', 'PIK3R1', 'FUBP1',	
                                       'RB1', 'NOTCH1', 'BCOR', 'CSMD3',
                                       'SMARCA4', 'GRIN2A',	'IDH2', 'FAT4',	'PDGFRA'
                                       ],
                 features_to_keep=[], features_to_drop=['Age_at_diagnosis'],
                 na_values=[], custom_preprocessing=default_preprocessing,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.

            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... categorical_features=['workclass', 'education', 'marital-status',
            ... 'occupation', 'relationship', 'native-country', 'race'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Note that we are now adding `race` as a `categorical_features`.
            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

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


        super(GliomaDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
