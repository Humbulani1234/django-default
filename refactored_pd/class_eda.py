"""
	==============================
	MCAR adhoc tests vs MNAR, MAR
	==============================

	======
	Plots
	======

"""

import ED
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

import class_base
import class_modelperf
import class_traintest

# --------------------------------------------------------------------Class EDA----------------------------------------------------


class EDA(Base):

	def __init__(self):

	def Point_Biserial_Test_Binary(independent, target):
	    
	    ''' Point Biserial Test for Binary vs Numerical varaibales '''
	    
	    df_loan_float_po_bi = pd.concat([independent, target], axis=1)
	    df_loan_float_po_bi_GB = df_loan_float_po_bi.groupby(target.name)
	    
	    df_0 = df_loan_float_po_bi_GB.get_group(0)
	    k = df_0[independent.name].mean()
	    
	    df_1 = df_loan_float_po_bi_GB.get_group(1)
	    j = df_1[independent.name].mean()
	    
	    standard_dev_AGE = df_loan_float_po_bi[independent.name].std()
	    mean_AGE = df_loan_float_po_bi[independent.name].mean()

	    proportion_GB_1 = df_1[target.name].count()/df_loan_float_po_bi[target.name].count()
	    proportion_GB_0 = 1-proportion_GB_1

	    r_po_bi = ((j-k)/standard_dev_AGE)*sqrt((independent.shape[0]*proportion_GB_0)\
	               *(1-proportion_GB_0)/(independent.shape[0]-1))

	    t_po_bi = r_po_bi/sqrt((1-r_po_bi**2))
	    
	    return t_po_bi, 
	

	def Chi_Square_Categorical_Test(independent, target):
	    
	    ''' Chi Square test for Categorical Variables '''

	    h_chi = pd.crosstab(independent, target)
	    chi_val, p_val, dof, expected = chi2_contingency(h_chi)
	    return chi_val, p_val

	def Chi_Square_Missingness_Categorical_Test(independent, target):
	    
	    '''Missing variables Test - Adhoc'''
	    
	    missingness = independent.isnull()
	    h_chi = pd.crosstab(missingness, target)
	    chi_val, p_val, dof, expected = chi2_contingency(h_chi)
	    
	    return chi_val, p_val


	def Pearson_Correlation_Test(independent, target):
	    
	    ''' Pearson correlation test function '''
	    
	    pearson_coef, p_value = stats.pearsonr(independent, target)
	    return pearson_coef, p_val


	def Correlation_Value_Greater(corr_threshold, dataframe):

		""" Multicollinearity investigation """
	    
	    dataframe_corr = Correlation_Plot(dataframe)
	    g = []

	    for i in range(dataframe_corr.shape[0]):
	        for j in range(dataframe_corr.shape[0]):
	            
	            if (dataframe_corr.iloc[i,j]>corr) or (dataframe_corr.iloc[i,j]<-corr):
	                g.append(dataframe_corr.iloc[i,j])
	    
	    return g        

	def get_Indexes(dataframe, value):
	    
	    list_of_pos = []
	    result = dataframe.isin([value])
	    series_obj = result.any()
	    column_names = list(series_obj[series_obj == True].index)
	    
	    for col in column_names:    
	        rows = list(result[col][result[col] == True].index)
	        
	        for row in rows:
	            list_of_pos.append((row,col))
	    
	    return list_of_pos


	def get_Variables_for_Corr_Greater(corr_thereshold, dataframe, value):
	       
	    dataframe_corr = Correlation_Plot(dataframe)
	    g_1 = Correlation_Value_Greater(corr_thereshold, dataframe)
	    list_of_pos_1 = get_Indexes(dataframe, value)

	    u = []

	    for i in g_1:
	        t = getIndexes(dataframe_corr, i)
	        u.append([item for item in t if item[0]!=item[1]])
	        
	    return u


	def VIF_value(dataframe, target):
	    
	    '''Calculate variance inflation factor'''

	    ols = statsmodels.regression.linear_model.OLS(target, dataframe.drop(labels=[target.name]), axis=1)
	    res_ols = ols.fit()              
	    VIF = 1/(1-res_ols.rsquared_adj**2)
	    
	    return VIF