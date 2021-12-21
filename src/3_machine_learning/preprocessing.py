import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import pylab as py
import statsmodels.api as sm
import statsmodels.stats as sm_stats
import statsmodels.stats.api as sms
import math
from datetime import datetime, date
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

class MergeTransformer(TransformerMixin):
    """
    A tranformer for merging 2 datasets.
    """
    cols = list()
    
    def __init__(self, *args, **kwargs):
        """
        Initialize method.

        :param *args: 2 datasets are instance of pandas.core.frame.DataFrame
        :param **kwargs: dictionary of names of columns
        """
        self.data1 = args[0]
        self.data2 = args[1]
        for col in kwargs.values():
            self.cols.append(col)
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to merge 2 datasets in *args with columns in **kwargs.

        :param X: 2 datasets are instance of pandas.core.frame.DataFrame
        :returns: pandas.core.frame.DataFrame
        """
        X = self.data1.merge(self.data2, on=self.cols, how = 'inner')
        return X
    
class ScalingTransform(TransformerMixin):
    """
    A tranformer for normalising columns via scaling.
    """
    
    def __init__(self, data,  *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        """
        self.data = data
        self.cols = list(args)
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to delete columns specified in *args to be deleted from data.

        :param data: pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        :returns: pandas.core.frame.DataFrame without columns specified in *args
        :raises keyError: raises an exception in case column was not found
        """
        try:
            for col in self.cols:
                scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
                X[col]=scaler.fit_transform(X[col].values.reshape(-1,1))
                #X[: , col] = power_transform(X[: , col].values.reshape(-1,1))
            return X
        except KeyError:
            print(f'Something from {self.cols} not found in dataset!') 
    
    
class PowerTransform(TransformerMixin):
    """
    A tranformer for normalising columns via power transformation.
    """
    
    def __init__(self, data, method, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        """
        self.data = data
        self.cols = list(args)
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to delete columns specified in *args to be deleted from data.

        :param data: pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        :returns: pandas.core.frame.DataFrame without columns specified in *args
        :raises keyError: raises an exception in case column was not found
        """
        try:
            for col in self.cols:
                 X[col] = power_transform(X[col].values.reshape(-1,1))
                 #X[: , col] = power_transform(X[: , col].values.reshape(-1,1))
            return X
        except KeyError:
            print(f'Something from {self.cols} not found in dataset!') 
    
class DropOutliersTransformer(TransformerMixin):
    """
    A tranformer for dropping outliers from the dataset.
    """
    
    def __init__(self, data, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing columns with outliers
        """
        self.data = data
        self.cols = list(args)
                
    def fit(self, X, y=None):        
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Drops outliers in columns

        :param X: The dataset.
        :param: column: Feature of pandas dataframe
        """
        for column in self.cols:
            random_data_std = np.std(X[column])
            random_data_mean = np.mean(X[column])
            anomaly_cut_off = random_data_std * 3
            lower_limit  = random_data_mean - anomaly_cut_off 
            upper_limit = random_data_mean + anomaly_cut_off
            X.loc[X[column] < lower_limit, column] = random_data_mean - random_data_std * 2
            X.loc[X[column] > upper_limit, column] = random_data_mean + random_data_std * 2
            
        #for column in self.cols:
        #    if ((stats.skew(X[column]) < -2) or (stats.skew(X[column]) > 2)):
        #        X[column] = np.log(X[column]+ (-X[column].min()))
        #    X.loc[X[column] < X[column].quantile(.05), column] = X[column].quantile(.05)
        #    X.loc[X[column] > X[column].quantile(.95), column] = X[column].quantile(.95) 
        return X
    
class RatioFillingTransformer(TransformerMixin):
    """
    A tranformer for ratio.
    """
    
    def __init__(self,  data, colToFill, corrCol, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param colToFill: The feature with NaN values
        :param corrCol: The feature which correlates with the parameter colToFill
        :param *args: optional
        """
        self.data = data
        self.colToFill = colToFill
        self.corrCol = corrCol
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to transform column to a numeric and change a type.

        :param X: pandas.core.frame.DataFrame
        :returns: transformed pandas.core.frame.DataFrame
        :raises keyError: raises an exception in case column was not found
        """
        try:
            ratio = X[self.colToFill]/X[self.corrCol]
            X[self.colToFill] = X[self.colToFill].fillna(X[self.corrCol] * ratio.mean())
            return X
        except KeyError:
            print(f'Something from {self.what} not found in dataset!')  
            
            
class FillingTransform(TransformerMixin):
    """
    A tranformer for filling NaN values in columns.
    """
    
    def __init__(self, data, method, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        """
        self.data = data
        self.cols = list(args)
        self.method = method
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to delete columns specified in *args to be deleted from data.

        :param data: pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        :returns: pandas.core.frame.DataFrame without columns specified in *args
        :raises keyError: raises an exception in case column was not found
        """
        try:
            if self.method == "mean":
                for col in self.cols:
                    X[col] = X[col].fillna(X[col].mean())
                return X
            if self.method == "median":
                for col in self.cols:
                    X[col] = X[col].fillna(X[col].median())
                return X
            if self.method == "knn":
                for col in self.cols:
                    imputer = KNNImputer(n_neighbors=5)
                    imputer.fit(X[col].values.reshape(-1,1))
                    Xtrans = imputer.transform(X[col].values.reshape(-1,1))
                    X[col] = pd.DataFrame(Xtrans)
                return X
            if self.method == "regression":
                for col in self.cols:
                    X[col] = X[col].interpolate(method='linear')
                return X
            
            
        except KeyError:
            print(f'Something from {self.cols} not found in dataset!') 
    
class GetNumericalTransformer(TransformerMixin):
    """
    A tranformer for getting numerical values from the dataset.
    """
    
    def __init__(self, data, col, what, category=None, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param col: col to be transformed
        :param what: dictionary, transform based on this
        :param category: change type based on this
        :param *args: optional
        """
        self.data = data
        self.col = col
        self.what = what
        self.category = category
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform_age(self, born):
        """
        Computes current age.

        :param born: Date of birth 
        returns: Current age
        """
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    
    def transform(self, X, **transform_params):
        """
        Function to transform column to a numeric and change a type.

        :param X: pandas.core.frame.DataFrame
        :returns: transformed pandas.core.frame.DataFrame
        :raises keyError: raises an exception in case column was not found
        """
        try:
            if self.col != 'birthdate':
                X[self.col].replace(self.what, inplace=True)
                if self.category is not None:
                    X = X.astype({self.col: self.category})
            else:
                X = X.astype({self.col: self.category})             # get birthdate in same format 
                X[self.col] = X[self.col].apply(self.transform_age) # get age instead of birthdate
                X = X.rename(columns = {self.col: 'age'})           # rename col birthdate as age
            return X
        except KeyError:
            print(f'Something from {self.what} not found in dataset!')  
            
class DropColsTransformer(TransformerMixin):
    """
    A tranformer for droping columns from the dataset.
    """
    
    def __init__(self, data, *args):
        """
        Initialize method.

        :param data: The dataset is instance of pandas.core.frame.DataFrame
        :param *args: tuple of *str, specifing cols to be dropped
        """
        self.data = data
        self.cols = list(args)
        
    def fit(self, X, y=None):
        """
        Fits transformer over data.

        :param X: The dataset to pass to the transformer.
        :returns: The transformer.
        """
        return self
    
    def transform(self, X, **transform_params):
        """
        Function to delete columns specified in *args to be deleted from data.

        :param X: pandas.core.frame.DataFrame
        :returns: pandas.core.frame.DataFrame without columns specified in *args
        :raises keyError: raises an exception in case column was not found
        """
        try:
            X = X.drop(self.cols, axis=1)
            return X
        except KeyError:
            print(f'Something from {self.cols} not found in dataset!') 
            return X
        

    
class OneRBestAtribute(object):
    """
    1R implementation
    """
    
    def __init__(self, data):
        """
        Constructor
        """
        self.data = data
        self.d_accuracy = dict()
        self.d_precision = dict()
        self.d_recall = dict()
        self.max_key = 0
    
    def fit(self, data):
        """
        Find best parameter
        
        :param data: dataset
        :param target: target
        """
        healthy_sample = data.loc[data.indicator == 1]
        sick_sample = data.loc[data.indicator == 0]

        # najdeme prieniky medzi zdravymi a chorymi pacientmi
        d_values = dict()
        for i in data:        
            d_values[str(i)] = list(
                set(healthy_sample[str(i)]) & set(sick_sample[str(i)])
            )
            
        target = data['indicator']

        # ziskame najlepssi accuracy pre kazdy atribut
        for index, (k, v) in enumerate(d_values.items()):
            #print(k)
            if (str(k) == "indicator"):
                continue
            list_accuracy = list()
            list_precision = list()
            list_recall = list()
            for i in v:
                rules = dict()
                rules["total"] = target.count()
                rules["tp"] = len(data.loc[(data[str(k)] >= round(i, 3)) & (target == 1.0), 'indicator'])
                rules["tn"] = len(data.loc[(data[str(k)] < round(i, 3)) & (target == 0.0), 'indicator'])
                rules["fp"] = len(data.loc[(data[str(k)] >= round(i, 3)) & (target == 0.0), 'indicator'])
                rules["fn"] = len(data.loc[(data[str(k)] < round(i, 3)) & (target == 1.0), 'indicator'])
                list_accuracy.append((rules["tp"] + rules["tn"]) / rules["total"])
                list_precision.append(rules["tp"] / (rules["fp"] + rules["tp"]))
                list_recall.append(rules["tp"] / (rules["fn"] + rules["tp"]))
            max_accurancy = max(list_accuracy)
            max_index = list_accuracy.index(max_accurancy) # najdi index
            self.d_accuracy[str(k)] =  list_accuracy[max_index] # save max number
            self.d_precision[str(k)] = list_precision[max_index]
            self.d_recall[str(k)] = list_recall[max_index] 
            self.max_key = max(self.d_accuracy, key=self.d_accuracy.get)

            
    def __repr__(self):
        """
        Returns representation when printing an object
        """
        txt = "Najlepší atribút pre údaje je: " + self.max_key;
        txt += "\n\ts Accuracy: " + str(self.d_accuracy[self.max_key])
        txt += "\n\ts Precision: " + str(self.d_precision[self.max_key])
        txt += "\n\ts Recall: " + str(self.d_recall[self.max_key])
        return txt