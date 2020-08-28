import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
import sklearn.preprocessing
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer



class FeatureSelector(BaseEstimator, TransformerMixin):
    '''Custom transformer to extract columns passed as arguments'''
    
    def __init__(self, feature_names):
        '''Class constructor'''
        self._feature_names = feature_names
        
    def fit(self, features_df, target = None):
        '''Returns self and nothing else'''
        return self
    
    def transform( self, features_df, target = None):
        '''This method returns selected features'''
        return features_df[ self._feature_names]      
    


class DropMissing(BaseEstimator, TransformerMixin):
    '''Takes df, drops all missing
    '''
    def __init__(self, df):
        self.df = df
        
    def fit(self, df, target=None):
        '''Returns self, nothing else.
        '''
        return self
    
    def transform(self, df, target=None):
        df.dropna(axis=0, how='any', inplace=True)        
        return df
    
    

class CategoricalFeatsAdded(BaseEstimator, TransformerMixin):
    ''' A custom transformer to add engineered categorical features 
        Takes df, checks if 'national_inv' is negative and adds indicator variable
    '''
    def __init__ (self, neg_inv_balance=True, low_inventory=True, \
                  low_intransit=True, high_forecast=True):
        ''' class constructor'''
        self._neg_inv_balance = neg_inv_balance
        self._low_inventory = low_inventory
        self._low_intransit = low_intransit
        self._high_forecast = high_forecast
     
    def fit( self, features_df, target = None):
        ''' Returns self, nothing else is done here'''
        return self
    
    def transform(self, features_df, target = None):
        ''' Creates aformentioned features and drops redundant ones'''

        if self._neg_inv_balance:
            '''Check if needed'''
            features_df['neg_inv_balance'] = (features_df.national_inv < 0).astype(int) 

        if self._low_inventory:
            '''check if needed'''
            features_df['low_inventory'] = (features_df['national_inv'] < \
                                            features_df['national_inv'].median()).astype(int)
            
        if self._low_intransit:
            '''check if needed'''
            features_df['low_intransit'] = (features_df['in_transit_qty'] < \
                                            features_df['in_transit_qty'].mean()).astype(int)
            
        if self._high_forecast:
            '''check if needed'''
            features_df['high_forcast'] = (features_df['forecast_3_month'] > \
                                           features_df['forecast_3_month'].mean()).astype(int)

        return features_df.values

    
    
class RemoveNegativeValues(BaseEstimator, TransformerMixin):
    '''Takes df, converts all negative values to positive
    '''
    def __init__(self, features_df):
        self.features_df = features_df
        
    def fit(self, features_df, target=None):
        '''Returns self, does nothing else
        '''
        return self
    
    def transform(self, features_df, target=None):
        '''Takes df, returns absolute values
        '''
        features_df = features_df.abs()        
        return features_df


        
class CategoricalImputerTransformer(BaseEstimator, TransformerMixin):
    '''This transformer imputes missing values on categorical pipeline'''
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        
    def fit(self, features_df, target=None):
        return self
    
    def transform(self, features_df, target=None):
        imputer = SimpleImputer(missing_values = np.NaN,
                                strategy='most_frequent')
        
        # Fit data to the imputer object 
        imputer = imputer.fit(features_df)
        
        # Impute the data      
        imputed = imputer.transform(features_df)
        
        features_df = pd.DataFrame(data=imputed)
    
        return features_df
    
    
                
class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    '''This transformer imputes missing values'''
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        
    def fit(self, features_df, target=None):
        return self
    
    def transform(self, features_df, target=None):
        imputer = SimpleImputer(missing_values = np.NaN,
                                strategy='median')
        
        # Fit data to the imputer object 
        imputer = imputer.fit(features_df)
        
        # Impute the data      
        imputed = imputer.transform(features_df)
        
        features_df = pd.DataFrame(data=imputed)
    
        return features_df    
    
    
    
class CapOutliers(BaseEstimator, TransformerMixin):
    '''Takes df, caps outliers
    '''
    #capped_features = pd.DataFrame()
    
    def __init__(self, features_df):
        self.features_df = features_df
        
    def fit(self, features_df, Target=None):
        '''Returns self, does nothing else
        '''
        return self
    
    def transform(self, features_df, q=0.90, target=None):
        for col in features_df.columns:

            if (((features_df[col].dtype)=='float64') | ((features_df[col].dtype)=='int64')):
                percentiles = features_df[col].quantile([0.01,q]).values
                features_df[col][features_df[col] <= percentiles[0]] = percentiles[0]
                features_df[col][features_df[col] >= percentiles[1]] = percentiles[1]

            else:
                features_df[col]=features_df[col]
        
        return features_df
    

    
class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    ''' This transformer standardizes all numerical features'''
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        
    def fit(self, features_df, target=None):
        return self
    
    def transform(self, features_df, target=None):
        features = features_df
        scaler = StandardScaler().fit(features)
        
        features_df = scaler.transform(features)

        return features_df
    
    
    
class DelUnusedCols(BaseEstimator, TransformerMixin):
    '''This transformer deletes unused columns from a data pipeline
       Col 0 holds an extra column for 'national_inv' added through the categorical feats. pipeline.
       This row is no longer needed after new categorical features leveraging the column are engineered
    '''
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        
    def fit(self, features_df, target=None):
        return self
    
    def transform(self, features_df, target=None):
        a = features_df
        a = np.delete(a,[0,1,2],1)
        features_df = a
        return features_df
  
