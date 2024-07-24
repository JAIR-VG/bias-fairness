import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'neutral or dissatisfied', 0.0: 'satisfied'}],
    'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'}],
}

class AirlineDataset(StandardDataset):
    """Airline Dataset.
    Datasets taken from https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
    --------
    train dataset: 24 attributes + 1 label Class
    Attributes to be removed Number, Id
    Gender attribute mapped 1.0 = Male, 0.0 Female
    Female = 52,727 (priviliged)
    Male   = 51,177
    satisfied = Label Class
    Mapped: 1.0: 'neutral or dissatisfied', 0.0: 'satisfied'
    'neutral or dissatisfied' = 58,879 (Favorable Class) = 1
    satisfied = 45,025 = 0
    -------------
    Test Dataset
    Gender
    Female = 13,172 (privileged) = 1
    Male = 12,804 (unprivileged) = 0

    'neutral or dissatisfied' = 14,573 (Favorable Class)
    'satisfied' = 11, 403

    """

    def __init__(self, label_name='satisfaction',
                 favorable_classes=[1],
                 protected_attribute_names=['Gender'],
                 privileged_classes=[['Female']],
                 instance_weights_name=None,
                 categorical_features=['Customer_Type', 'Type_of_Travel',
                     'Class'],
                 features_to_keep=[], features_to_drop=['Number', 'id'],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
   
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data', 'airline', 'train.csv')
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data', 'airline', 'test.csv')
        #filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                          'data', 'airline', 'airline.csv')
        # as given by adult.names
       # column_names = ['Number', 'id',	'Gender', 'Customer_Type',	'Age',
       #                 'Type_of_Travel', 'Class', 'Flight_Distance', 'Inflight_wifi_service',
       #                 'Departure_Arrival_time_convenient', 'Ease_of_Online_booking',
       #                 'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',	
       #                 'Inflight_entertainment', 'On_board_service', 'Leg_room_service',	
       #                 'Baggage_handling', 'Checkin_service',	'Inflight_service', 'Cleanliness',	
       #                 'Departure_Delay', 'Arrival_Delay',	'satisfaction']
        try:
            train = pd.read_csv(train_path, header=0)
            test = pd.read_csv(test_path, header=0)
          
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the datasets from:")
            print("\n\thttps://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'airline'))))
            import sys
            sys.exit(1)

        df = pd.concat([train, test], axis=0)

        super(AirlineDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
