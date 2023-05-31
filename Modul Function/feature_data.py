from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class Scaling:
    """
    Class for scaling data, this class will change original data
    to data scaling
    """
    def __init__(self):
        pass
    
    def standardizeData(self, data, scaler=None):
        """
        This function is used to convert data to standardized scaler
        """
        if scaler is None:
            # Fit scaler
            scaler = StandardScaler()
            scaler.fit(data)

        # Transform data
        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

        return data_scaled, scaler

    def Minmax(self, data, minmax = None):
        """
        This function used for convert data to
        Minmax scaler
        """
        if minmax == None:
            # Fit scaler
            minmax = MinMaxScaler()
            minmax.fit(data)

        # Tranform data
        data_minmax = minmax.transform(data)
        data_minmax = pd.DataFrame(data_minmax,
                                   index = data.index,
                                   columns = data.columns)

        return data_minmax, minmax  

    def Normalizer(self, data):
        """
        This function used for convert data to
        Normal scaler
        """
        # make scaler
        normal = Normalizer()

        # normalize the data (fit)
        normalized_data = normal.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_data,
                               index = data.index,
                               columns = data.columns)

        return normalized_data, normal

class Transform:    
    """
    Class for transform data, this class will change original data
    to data transform
    """
    def __init__(self):
        pass 
 
    def log(self, data) :
        """
        This function used for convert data to
        Transformation log
        """
        data_log = np.log1p(data)
        data_log = pd.DataFrame(data_log,
                           index = data.index,
                           columns = data.columns)
        return data_log
    
    def yeo(self, data) :
        """
        This function used for convert data to
        PowerTransformation yeo-johnson
        """    
        pt = PowerTransformer(method='yeo-johnson')
        data_yeo = pt.fit_transform(data)
        data_yeo = pd.DataFrame(data_yeo,
                       index = data.index,
                       columns = data.columns)
        return data_yeo
    
    def box(self, data) :
        """
        This function used for convert data to
        PowerTransformation box-cox
        """    
        pt = PowerTransformer(method='box-cox', standardize=False)
        data_box = pt.fit_transform(data + 0.0001)
        data_box = pd.DataFrame(data_box,
                       index = data.index,
                       columns = data.columns)
        return data_box

class Reduction:    
    """
    Class for transform data, this class will change original data
    to data transform
    """
    def __init__(self):
        pass 

    def dimension_pca(self, data, n_components):
        """
        This function used for convert data to Dimensionality reduction PCA
        """
        if n_components > data.shape[1]:
            n_components = data.shape[1]

        pca = PCA(n_components=n_components)

        # Fit the PCA model to the data
        pca.fit(data)

        # Transform the data using the fitted PCA model
        pca_data = pca.transform(data)
        pca_data = pd.DataFrame(pca_data,
                               index = data.index,
                               columns = ['PC{}'.format(i) for i in range(1, n_components+1)])

        return pca_data

class imbalance:
    """
    This Class function for making imbalanced data to balanced
    using SMOTE, Oversampling, and Undersampling
    """
    def __init__(self):
        pass 
    
    def perform_smote(self, X_train, y_train):
        """
        SMOTE works by creating a synthetic sample for the 
        minority class by taking some of the nearest neighbors of 
        the minority class and building a synthetic sample between those points.
        """
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        
        return X_train_res, y_train_res
  
    def perform_undersampling(self, X_train, y_train, random_state=2):
        """
        reduce the number of samples from the majority class in the dataset so 
        that it is balanced by the number of samples from the minority class.
        """
        under_sampler = RandomUnderSampler(random_state=random_state)
        X_train_res, y_train_res = under_sampler.fit_resample(X_train, y_train)

        return X_train_res, y_train_res

    def perform_oversampling(self, X_train, y_train, random_state=2):
        """
        The way oversampling works is by duplicating samples from the minority class so that the number 
        of samples in the minority class is the same as the number of samples in the majority class
        """
        over_sampler = RandomOverSampler(random_state=random_state)
        X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)

        return X_train_res, y_train_res

    
