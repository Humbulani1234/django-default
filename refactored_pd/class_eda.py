
"""
    ==============================
    MCAR adhoc tests vs MNAR, MAR
    ==============================

    ======
    Plots
    ======
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

from class_plotting import Charts

class ExploratoryDataAnalysis(Charts, object):

    def __init__(self, independent: pd.DataFrame, target: pd.DataFrame):

        self.independent = independent
        self.target = target

    def __str__(self):
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ImputationCat.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def point_biserial(self):
        
        """ Point Biserial Test for Binary vs Numerical varaibales """
        
        df_loan_float_po_bi = pd.concat([self.independent, self.target], axis=1)
        df_loan_float_po_bi_GB = df_loan_float_po_bi.groupby(self.target.name)        
        df_0 = df_loan_float_po_bi_GB.get_group(0)
        k = df_0[self.independent.name].mean()        
        df_1 = df_loan_float_po_bi_GB.get_group(1)
        j = df_1[self.independent.name].mean()        
        standard_dev_AGE = df_loan_float_po_bi[self.independent.name].std()
        mean_AGE = df_loan_float_po_bi[self.independent.name].mean()
        proportion_GB_1 = df_1[self.target.name].count()/df_loan_float_po_bi[self.target.name].count()
        proportion_GB_0 = 1-proportion_GB_1
        r_po_bi = ((j-k)/standard_dev_AGE)*sqrt((self.independent.shape[0]*proportion_GB_0)\
                   *(1-proportion_GB_0)/(self.independent.shape[0]-1))
        t_po_bi = r_po_bi/sqrt((1-r_po_bi**2))
        
        return t_po_bi,     

    def chi_square_categorical_cat_test(self):
        
        """ Chi Square test for Categorical Variables """

        h_chi = pd.crosstab(self.independent, self.target)
        chi_val, p_val, dof, expected = chi2_contingency(h_chi)

        return chi_val, p_val

    def chi_square_missing_cat_test(self):
        
        """ Missing variables Test - Adhoc """
        
        missingness = self.independent.isnull()
        h_chi = pd.crosstab(missingness, self.target)
        chi_val, p_val, dof, expected = chi2_contingency(h_chi)
        
        return chi_val, p_val

    def pearson_corr_test(self):
        
        """ Pearson correlation test function """
        
        pearson_coef, p_value = stats.pearsonr(self.independent, self.target)

        return pearson_coef, p_val

    def _corr_value_greater(self, corr_threshold):

        """ Multicollinearity investigation """
        
        dataframe_corr = super(ExploratoryDataAnalysis,self).correlation_plot(self.dataframe)
        g = []
        for i in range(dataframe_corr.shape[0]):
            for j in range(dataframe_corr.shape[0]):                
                if (dataframe_corr.iloc[i,j]>corr) or (dataframe_corr.iloc[i,j]<-corr):
                    g.append(dataframe_corr.iloc[i,j])
        
        return g        

    def _get_indexes(self, value):
        
        list_of_pos = []
        result = self.dataframe.isin([value])
        series_obj = result.any()
        column_names = list(series_obj[series_obj == True].index)    
        for col in column_names:    
            rows = list(result[col][result[col] == True].index)            
            for row in rows:
                list_of_pos.append((row,col))
        
        return list_of_pos

    def get_var_corr_greater(self, corr_thereshold, value):
           
        dataframe_corr = super(ExploratoryDataAnalysis,self).corr_plot(self.dataframe)
        g_1 = self._corr_value_greater(corr_thereshold, self.dataframe)
        list_of_pos_1 = self._get_indexes(self.dataframe, value)
        u = []
        for i in g_1:
            t = self._getindexes(dataframe_corr, i)
            u.append([item for item in t if item[0]!=item[1]])
            
        return u

    def vif_value(self):
        
        '''Calculate variance inflation factor'''

        ols = statsmodels.regression.linear_model.OLS(self.target, self.dataframe.
                                                      drop(labels=[self.target.name]), axis=1)
        res_ols = ols.fit()              
        VIF = 1/(1-res_ols.rsquared_adj**2)
        
        return VIF