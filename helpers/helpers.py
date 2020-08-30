import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from numpy import percentile

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


"""
A set of helper function for the project
"""

def read_in_dataset(dset, verbose=False):
    """
    Read in one of the datasets 

    Keyword arguments:
    dset -- a string in {train_target, train_features}
    verbose -- whether or not to print info about the dataset

    Returns:
    A pandas dataframe

    """
    df = pd.read_csv(r'unzipped_data/{0}.csv'.format(dset))    

    if verbose:
        print('\n{0:*^80}'.format(' Reading in the {0} dataset '.format (dset)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns'))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' The first five rows look like this '))
        print(df.head())
        print('\n{0:*^80}\n'.format(' Below are data types for each column '))
        print(df.dtypes)
    
    return df

def read_in_excel(dset, verbose=False):
    """
        Read in one of the datasets 

        Keyword arguments:
        dset -- a string in {train_target, train_features}
        verbose -- whether or not to print info about the dataset

        Returns:
        A pandas dataframe
    """
    df = pd.read_excel(r'unzipped_data/{0}.xlsx'.format(dset))
    
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the {0} dataset '.format (dset)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns'))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' The first five rows look like this '))
        print(df.head())
        print('\n{0:*^80}\n'.format(' Below are data types for each column '))
        print(df.dtypes)
        
    return df


def merge_dataset(electronics, profiles):

    """
    Merging the electronics and profile datasets. Both need to have a common key clients

    Keyword arguments:
    electronics -- the dataframe of electronics data
    profiles -- the dataframe of profiles

    Returns:
    A pandas dataframe
    """
    train_data_merged = electronics.merge(profiles, how='right', on ='clients')

    return train_data_merged


def mergeFeatures():
    '''Merges sparce categorical features and renaming UPPER to lowercase for readability ease.
    '''
    # Merge'BUSINESS' and 'ENGINEERING' to 'BUS OR ENG'
    df.major.replace(['BUSINESS','ENGINEERING'], 'busOrEng', inplace=True)

    # Merge 'PHYSICS','CHEMISTRY','BIOLOGY' to 'sci'
    df.major.replace(['PHYSICS','CHEMISTRY','BIOLOGY'], 'sci', inplace=True)

    # 'LITERATURE' should be 'literature'
    df.major.replace('LITERATURE', 'literature', inplace=True)

    # 'MATH' should be 'math'
    df.major.replace('MATH', 'math', inplace=True)

    # 'COMPSCI' should be 'compsci'
    df.major.replace('COMPSCI', 'compsci', inplace=True)



def add_lessThanHighSchoolInd():
    '''Adds an indicator showing at least high school completed'''
    df['less_than_HighSchool'] = ((df.degree == 'NONE') & (df.major == 'NONE'))



def removeZeroSalaries():
    '''Removes all observations with zero salaries'''
    
    global df #Assigns 'df' as a global variable
    
    #Creating a series mask to remove all observations with salary == 0
    mask = df.salary
    series_mask = mask > 0
    mask[series_mask]

    # Remove all outliers with salary == 0
    df = df[df.salary > 0]
    
    
class ColumnTypes():
    #@staticmethod
    '''Selects numerical and categorical columns
    '''
    #Class Variables
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float']
    numdf = pd.DataFrame() #Empty dataframe to hold numerical columns
    non_numdf = pd.DataFrame() #Empty dataframe to hold non-numeric columns
    numerical_columns = []
    non_numerical_columns = []
    
    def __init__(self, df):
        self.df = df
        
    def histograms_numeric_columns(df):
        '''Takes df, numerical columns as list
           Returns a group of histagrams
        '''
        ColumnTypes.numdf= df.select_dtypes(include=ColumnTypes.numerics)
        ColumnTypes.non_numdf = df.select_dtypes(exclude=ColumnTypes.numerics)
        
        #Create numerical columns list 
        for col in ColumnTypes.numdf.columns:
            ColumnTypes.numerical_columns.append(col)
            
        #Create categorical columns list
        for col in ColumnTypes.non_numdf.columns:
            ColumnTypes.non_numerical_columns.append(col)
            
        #Delete unused dataframes
        del ColumnTypes.numdf
        del ColumnTypes.non_numdf
    
        #plotting
        f = pd.melt(df, value_vars=ColumnTypes.numerical_columns) 
        g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)        
        g = g.map(sns.distplot, 'value')
        return g
    
def  check_missing(df):
    '''Takes df, checks null
    
    Args: dataframe name
    '''
    if df.isnull().sum().any() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False)
        total = mask_total[mask_total > 0]
        
        mask_percent = df.isnull().mean().sort_values(ascending=False)
        percent = mask_percent[mask_percent > 0]
    
        missing_data = pd.concat([total, percent], axis=1,keys=['Total', 'Percent'])
    
        #print(f'Total and Percentage of Missing Values Found:\n {missing_data}')
        print('\n{0:*^80}\n'.format(f' Total and Percentage of Missing Values Found '))
        print(missing_data)
    
    else:
         print('No Missing Values found.')


    

def plot_missing(df):
    '''Takes df, return bar plot of features with missing values
    '''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
            

        
        
def view_many_missing(df, missing_percent):
    '''Takes df, Sepecified Missing Percent of total
    
    Returns:
         Columns of missing as list
    '''
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent]
    columns = series.index.to_list()
    print('\n{0:*^80}\n'.format(f' Features with over {missing_percent} percent missing '))
    return columns




def view_missing_by_row(df, n=0):
    '''Takes df, Returns rows with missing values
    
    Args:
        Dataframe
        n - Threshold of missing count if specified (Default 0, returning all)
    '''
    #Get value count of row with at least 1 missing value
    width = df.shape[1]
    value_count = df.count(axis=1)
    #filtered = len([count for count in value_count if count < 27])
    filtered = len(value_count[value_count<27])
    
    if filtered != 0:
        print('\n{0:*^80}\n'.format(f' {filtered} rows with missing data found in dataset'))
                        
        #Calculate missing values count by row
        missing = df.shape[1] - value_count
      
        #append missing to main df
        df = df.merge(missing.rename('missing'), left_index=True, right_index=True)
    
        #Get datfarame of all observations with 'n' missing records
        df = df[df.missing == n]
    
        #View observations with n records missing    
        print(f' {df.shape[0]} Rows with {n} records missing found')
        return df
    
    else:
        print('\n{0:*^80}\n'.format(f' No rows with {n} missing data found in dataset'))
                        

            
def drop_columns_w_many_nans(df, missing_percent):
    '''Takes df, missing percentage
       Drops columns with missing values exceeding specified percent
       
     Returns:
        df
    '''
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent]
    list_of_cols = series.index.to_list()
    df = df.drop(columns=list_of_cols, inplace = True)
    print('\n{0:*^80}\n'.format(f' Columns dropped '))
    print(list_of_cols)
    #print(f'Columns droped: \n {list_of_cols}')
    #return df #note need to fix this function as not updating datfarame after dropping specified column (fixed, added inplace = True)
    

    
    
def check_non_unique(df):
    '''Takes df, checks for non unique columns
    
    Args:
        Dataframe
    '''
    mask = dict(df.nunique().sort_values(ascending=False))
    counter = 0
    for (col, count) in mask.items():
        if count == 0:
            counter += 1
                    
    if counter > 0:
        print('\n{0:*^80}\n'.format(' The data has the following non unique columns'))
        for (k,v) in mask.items():
            if v == 0:
                print(k)
    else:
        print('No columns with non unique values found')
        


        
        
def check_dups(df):
    '''Takes df, checks for duplicate rows
    
    Args: 
       Dataframe
    '''
    #get data labels
    mask = dict(df.nunique().sort_values(ascending=False))
    labels = []
    
    for (k,v) in mask.items():
        if v == len(df):
            labels.append(k) #Assumes label if column has non unique values
        else:
            break
    
    
    if labels == []:
        print('Data frame has no unique rows ... \n Check data labels')

    else:
        #check duplicates
        for name in labels:
            if (df[name].nunique() == len(df)) == False: #Checks if there is 1 row per record
                duplicateRowsDF = df[df.duplicated()]
                print("Duplicate Rows except first occurrence based on all columns are :")
                print(duplicateRowsDF)
            else:
                print('No duplicates found')
          
        
        
        

def move_to_first(df, col_name):
    '''Takes column name, moves to first column
    '''
    first_col = df.pop(col_name)
    df.insert(0, col_name, first_col)
    
    
        

def move_to_last(df, n, col_name):
    '''Takes column name, col index to move to -1, moves to first column
      Example:
          for dataframe of shape 1000,10, to move 9th column to last column
          use n = 10-1, which is 9
    '''
    first_col = df.pop(col_name)
    df.insert(n, col_name, first_col)
    
    
    
        
def move_column(df, col_name,col_ix):
    '''Takes df, column name, column index and moves to selected column index position
    '''
    move_col = df.pop(col_name)
    df.insert(col_ix, col_name, move_col)
                    

        
        
class CheckCardinality:
    '''Methods to identify sparce classes
    '''
    #class variables
    name_df = pd.DataFrame()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float']
    #obj_df = df.select_dtypes(exclude=numerics)
    obj_df = pd.DataFrame()
    #mask = dict(obj_df.nunique().sort_values(ascending=False))
    mask = {}
    counter = 0
    sparse_df = pd.DataFrame()    
    
    def __init__(self, df, threshold, plots):
        self.df = self
        self.threshold = threshold
        self.plots = plots
        
    def check_sparse_classes(df, threshold = 0.05, plots=False):
        '''Takes df, checks for sparse classes
        Args:
            Dataframe
            threshold (float):
                Least percent of max class count a class should have to not be considered sparse
                Class size calculated by class_max_count * threshold
                Use lower thresholds for small datasets and vice versa
                Default 5%             
        '''
        #mask = dict(obj_df.nunique().sort_values(ascending=False))
        CheckCardinality.obj_df = df.select_dtypes(exclude=CheckCardinality.numerics)
        CheckCardinality.mask = (CheckCardinality.obj_df.nunique().sort_values(ascending=False))
        
        #Get all column names of type = object
        for (col, count) in CheckCardinality.mask.items():
            CheckCardinality.sparse_df = CheckCardinality.sparse_df.append({'column':col}, ignore_index=True)
            
        #create dict for every name in sparse_df
        for name in CheckCardinality.sparse_df.column:
            
            name_dict = dict(CheckCardinality.obj_df[name].value_counts())
            
            for (k,v) in name_dict.items():
                
                #Test value count meets threshold
                if v <= threshold * (max(CheckCardinality.obj_df[col].value_counts())):
                    CheckCardinality.counter = CheckCardinality.counter + 1
                    
                    #Append sparse class and value count to name_df
                    CheckCardinality.name_df = CheckCardinality.name_df.append({'column': name, 'class': k, 'count': int(v)}, ignore_index = True)
                    
        if CheckCardinality.counter != 0:
            print('{0:*^80}\n'.format(' Sparse features found in dataset'))
            print(CheckCardinality.name_df)
            
        else:
            print(f' No Sparse classes found in dataset at a {threshold} threshold')
        
        if np.logical_and(plots, CheckCardinality.counter != 0):
            #If True: Returns distributions of sparse classes, default None
            print('\n{0:*^80}\n'.format(' Sparse class distributions '))
            for col in CheckCardinality.name_df.column.unique():
                print (col)
                sns.countplot(y= col, data=df)
                plt.show()




def heatmap_numeric_w_dependent_variable(df, method, target_variable):
    '''
    Takes df, method (pearson,kendall,or spearman)a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df.corr(method=method)[[target_variable]].sort_values(by=target_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g    
    
    
    
def get_multicollinear_feats(df, target):
    '''Takes df, returns multicorrelated features
    '''
    correlated_features = set()
    correlation_matrix = df.drop(target, axis=1).corr()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    return correlated_features
    
    
    
    
def visualize_outlier_regions(target, df):
    '''Takes dataframe and target column
       Uses sklearns IsolationForest lib to identify outlier regions
    
      Returns:
          A visualization highlighting the regions where the outliers fall.         
    '''
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df[target].values.reshape(-1, 1))
    
    xx = np.linspace(df[target].min(), df[target].max(), len(df)).reshape(-1,1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)
    
    plt.figure(figsize=(10,4))
    plt.plot(xx, anomaly_score, label='anomaly score')

    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
    plt.legend()
    plt.ylabel('anomaly score')
    plt.xlabel(target)
    plt.show();    
    


    
class CheckOutliers:
    '''Holds helper functions to identify outliers
    '''
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    
    def __init__(self, dataframe, cols, outliers_fraction, standardize):
        self.dataframe = dataframe
        self.cols = cols
        self.outliers_fraction = outliers_fraction
        self.standardize = standardize
        
    def get_CBOLF_scores(dataframe, cols, outliers_fraction = 0.01):
        '''Takes df, a list selected column nmaes, outliers_fraction = 0.01 default
        
        Returns:
            df with CBOLF scores added         
        '''
        #standardize selected variables
        minmax = MinMaxScaler(feature_range=(0,1))
        dataframe[cols] = minmax.fit_transform(dataframe[cols])
        
        #Convert dataframe to a numpy array in order to incorprate our algorithm
        arrays = []
        for row in cols:
            row = dataframe[row].values.reshape(-1,1)
            arrays.append(row)
        X = np.concatenate((arrays),axis=1)
            
        #fit
        clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=0)
        clf.fit(X)
            
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1
            
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
            
        CheckOutliers.df1 = dataframe
        CheckOutliers.df1['outlier'] = y_pred.tolist()
            
        print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with CBLOF')
        #return CheckOutliers.df1    
    
    def get_HBOS_scores(dataframe, cols, outliers_fraction = 0.01, standardize=True):
        '''Takes df, a list selected column nmaes, outliers_fraction = 0.01 default
        
        Returns:
            df with CBOLF scores added         
        '''
        if standardize:
            #standardize selected variables
            minmax = MinMaxScaler(feature_range=(0,1))
            dataframe[cols] = minmax.fit_transform(dataframe[cols])
        
        #Convert dataframe to a numpy array in order to incorprate our algorithm
        arrays = []
        for row in cols:
            row = dataframe[row].values.reshape(-1,1)
            arrays.append(row)
        X = np.concatenate((arrays),axis=1)
            
        #fit
        clf = HBOS(contamination=outliers_fraction)
        #clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=0)
        clf.fit(X)
            
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1
            
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
            
        CheckOutliers.df2 = dataframe
        CheckOutliers.df2['outlier'] = y_pred.tolist()
            
        print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with HBOS')
        #return CheckOutliers.df1  
    
    def get_IF_scores(dataframe, cols, outliers_fraction = 0.01, standardize=True):
        '''Takes df, a list selected column nmaes, outliers_fraction = 0.01 default
        
        Returns:
            df with Isolation Forest (IF) scores added         
        '''
        if standardize:
            #standardize selected variables
            minmax = MinMaxScaler(feature_range=(0,1))
            dataframe[cols] = minmax.fit_transform(dataframe[cols])
        
        #Convert dataframe to a numpy array in order to incorprate our algorithm
        arrays = []
        for row in cols:
            row = dataframe[row].values.reshape(-1,1)
            arrays.append(row)
        X = np.concatenate((arrays),axis=1)
            
        #fit
        clf = IForest(contamination=outliers_fraction,random_state=0)
        clf.fit(X)
            
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1
            
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
            
        CheckOutliers.df3 = dataframe
        CheckOutliers.df3['outlier'] = y_pred.tolist()
            
        print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with HBOS')
        #return CheckOutliers.df3  
    
    def get_KNN_scores(dataframe, cols, outliers_fraction = 0.01, standardize=True):
        '''Takes df, a list selected column nmaes, outliers_fraction = 0.01 default
        
        Returns:
            df with KNN scores added         
        '''
        if standardize:
            #standardize selected variables
            minmax = MinMaxScaler(feature_range=(0,1))
            dataframe[cols] = minmax.fit_transform(dataframe[cols])
        
        #Convert dataframe to a numpy array in order to incorprate our algorithm
        arrays = []
        for row in cols:
            row = dataframe[row].values.reshape(-1,1)
            arrays.append(row)
        X = np.concatenate((arrays),axis=1)
            
        #fit
        clf = KNN(contamination=outliers_fraction)
        clf.fit(X)
            
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1
            
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
            
        CheckOutliers.df4 = dataframe
        CheckOutliers.df4['outlier'] = y_pred.tolist()
            
        print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with KNN')
        #return CheckOutliers.df4      
  
    

    
    
def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames.
    
        see: https://hackersandslackers.com/compare-rows-pandas-dataframes/
    """
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    #diff_df.to_csv('data/diff.csv')
    return diff_df 
    


    
    
class Outliers:
    '''Holds helper functions to identify outliers
    '''
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    merged_df = pd.DataFrame()
    outlier_indices = []
    #row_id = None
    #name = 'client_id'
    n = None
    
    def __init__(self, dataframe, cols, outliers_fraction, row_id, n, cbolf, hbos, iforest, knn):
        self.dataframe = dataframe
        self.cols = cols
        self.outliers_fraction = outliers_fraction
        self.row_id = row_id
        self.n = n
        self.cbolf = cbolf
        self.hbos = hbos
        self.iforest = iforest
        self.knn = knn
        
    def get_outliers(dataframe, cols, outliers_fraction,row_id, n,
                     cbolf=True, hbos=True, iforest=True, knn=True):
        '''
        Params:
            row_id ('str'): unique row identifier on the dataframe
            n(int): Minimum number of timmes an observation should be flagged as an outlier 
                    to be considered one
        
        Retrurns:
            List of index labels for rows in the dataframe that are flagged as outliers
        
        '''
        #standardize selected numerical variables
        minmax = MinMaxScaler(feature_range=(0,1))
        dataframe[cols] = minmax.fit_transform(dataframe[cols])
        
        #Convert dataframe to a numpy array in order to incorprate our algorithm
        #Outliers.row_id = row_id
        arrays = []
        for row in cols:
            row = dataframe[row].values.reshape(-1,1)
            arrays.append(row)
        X = np.concatenate((arrays),axis=1)
               
        
        if cbolf:
            '''Runs Cluster-Based Outlier Local Factor (CBOLF)  algorithm to identify outliers'''
            #fit
            clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=0)
            clf.fit(X)
            
            #predict raw anomaly score
            scores_pred = clf.decision_function(X) * -1
            
            #prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            
            #Hold results to dataframe and print findings
            Outliers.df1 = dataframe
            Outliers.df1['outlier'] = y_pred.tolist()
            Outliers.df1 = Outliers.df1.loc[Outliers.df1['outlier'] == 1]
            print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with CBLOF')

        if hbos:
            '''Runs Histogram Based Outlier Score (HBOS) algorithm to identify outliers'''
            #fit
            clf = HBOS(contamination=outliers_fraction)
            clf.fit(X)
            
            #predict raw anomaly score
            scores_pred = clf.decision_function(X) * -1
            
            #prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            
            #Hold results to dataframe and print findings
            Outliers.df2 = dataframe
            Outliers.df2['outlier'] = y_pred.tolist()
            Outliers.df2 = Outliers.df2.loc[Outliers.df2['outlier'] == 1]
            print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with HBOS')
            
        if iforest:
            '''Runs Isolation Forest algorithm to identify outliers'''
            #fit
            clf = IForest(contamination=outliers_fraction,random_state=0)
            clf.fit(X)
            
            #predict raw anomaly score
            scores_pred = clf.decision_function(X) * -1
            
            #prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            
            #Hold results to dataframe and print findings
            Outliers.df3 = dataframe
            Outliers.df3['outlier'] = y_pred.tolist()
            Outliers.df3 = Outliers.df3.loc[Outliers.df3['outlier'] == 1]
            print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with IForest')
            
        if knn:
            '''Runs K-Nearest Neighbors algorithm to identify outliers'''
            #fit
            clf = KNN(contamination=outliers_fraction)
            clf.fit(X)
            
            #predict raw anomaly score
            scores_pred = clf.decision_function(X) * -1
            
            #prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            
            #Hold results to dataframe and print findings
            Outliers.df4 = dataframe
            Outliers.df4['outlier'] = y_pred.tolist()
            Outliers.df4 = Outliers.df4.loc[Outliers.df4['outlier'] == 1]
            print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, 'found with KNN')
            
        #Merge dataframes
        merged_df = pd.concat([Outliers.df1, Outliers.df2, Outliers.df3, Outliers.df4])
    
        #Get counts (Count number of times an observation is identified as an outlier)
        merged_df['count'] = merged_df.groupby(row_id)[row_id].transform('count')
        #outliers['count'] = outliers.groupby('client_id')['client_id'].transform('count')
    
        #Filter common outliers (Outlier identified by all n algorithms)
        #common = outliers.loc[outliers['count'] >= n]
        common = merged_df.loc[merged_df['count'] >= n]
    
        #drop duplicates
        common = common.drop_duplicates(keep='last')
    
        #get list of indices to be removed on main dataframe
        Outliers.outlier_indices = []
        for index in common.index:
            Outliers.outlier_indices.append(index)
                     
        #print(f' \n{common.shape[0]} outliers commonly found by all algorithms\n')
        print(f' \n{len(Outliers.outlier_indices)} outliers commonly found by all algorithms\n')
        print(f'The row index labels are:\n {Outliers.outlier_indices}')
        return Outliers.outlier_indices
      
    
    
    
    
