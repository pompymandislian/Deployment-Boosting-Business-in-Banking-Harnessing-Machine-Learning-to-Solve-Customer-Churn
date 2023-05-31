import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Encoder:
    """
    This class is used for one-hot encoding categorical data
    """
    def __init__(self, data):
        self.data = data
        
    def Ohe_encoder(self):
        """
        This function used for change data categori
        to numeric for process modeling
        """
        data_num = self.data.select_dtypes(include=['number'])
        data_cat = self.data.select_dtypes(include=['object'])

        ohe = pd.get_dummies(data_cat)
        # concat data X_train
        concat = pd.concat([data_num, ohe],
                           axis = 1)

        return concat
    
    def label_encoder(self, data):
        """
        This function is used for encoding categorical data with label encoding
        """
        le = LabelEncoder()
        
        # Fit and transform the data
        data_encoded = le.fit_transform(data)
        
        return data_encoded