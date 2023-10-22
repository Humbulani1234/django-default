

"""
     ==========================
     TRAIN AND TESTING SAMPLES
     ==========================

         1. One Hot Encoding
         2. Train and Testing sample split

     =================
     One Hot Encoding:
     =================
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import warnings

from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

class OneHotEncoding(Base, object):

    def __init__(self, custom_rcParams, df_nomiss_cat, type_):

        super(OneHotEncoding,self).__init__(custom_rcParams)
        self.df_nomiss_cat = df_nomiss_cat
        self.type = type_
    
    def __str__(self):
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in OneHotEncoding.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"    

    def onehot_encoding(self):
    
        '''One Hot Encoding Function
        Test that no matter how you write the code the dataframe has no missing values
        and contains zero and ones for categorical data'''

        if self.type == "machine":    
            encoded_dataframes = []
            for col in self.df_nomiss_cat.columns:
                y = pd.get_dummies(self.df_nomiss_cat[col]).astype(int)
                encoded_dataframes.append(y)
            df_cat_onehotenc = pd.concat(encoded_dataframes, axis = 1)
            return df_cat_onehotenc
        elif self.type == "statistics":        
            encoded_dataframes = []
            for col in self.df_nomiss_cat.columns:                
                y = pd.get_dummies(self.df_nomiss_cat[col]).astype(int)
                n = len(pd.unique(self.df_nomiss_cat[col])) 
                self.df_nomiss_cat_ = y.drop(y.columns[n-1], axis=1) 
                encoded_dataframes.append(self.df_nomiss_cat_)
            df_cat_onehotenc = pd.concat(encoded_dataframes, axis = 1)

            return df_cat_onehotenc

    def create_xy_frames(self, df_float, target):

        if self.type == "machine":
            df_cat = self.onehot_encoding()
            df_total_partition = pd.concat([df_float, df_cat], axis = 1)
            x = df_total_partition.drop(labels=[target.name], axis=1)
            y = df_total_partition[target.name]
            
            return x, y

        elif self.type == "statistics":
            df_cat = self.onehot_encoding()
            df_total_partition = pd.concat([df_float, df_cat], axis = 1)
            x = df_total_partition.drop(labels=[target.name], axis=1)
            y = df_total_partition[target.name]
            
            return x, y

    def split_xtrain_ytrain(self, df_float, target):
    
        x, y = self.create_xy_frames(df_float, target)
        x_train_pd, x_test_pd, y_train_pd, y_test_pd = train_test_split(x, y, test_size=0.3, random_state=42)
        x_train_pd = x_train_pd.drop(labels=["_freq_"], axis=1) # temp, for mach it has to be dropped
        x_test_pd = x_test_pd.drop(labels=["_freq_"], axis=1) # temp
   
        return x_train_pd, x_test_pd, y_train_pd, y_test_pd

    def train_val_test(self, df_float, target):

        """ This method creates the Trainning, Validation and Testing datasets """

        x_train_pd, x_split_pd, y_train_pd, y_split_pd = self.split_xtrain_ytrain(df_float, target)
        x_val_pd, x_test_pd, y_val_pd, y_test_pd = train_test_split(x_split_pd, y_split_pd, test_size=0.3, random_state=42)
        return x_train_pd, y_train_pd, x_val_pd, y_val_pd, x_test_pd, y_test_pd

