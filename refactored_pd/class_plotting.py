

# ==============================
# MCAR adhoc tests vs MNAR, MAR
# ==============================

# ======
# Plots
# ======

import ED
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import logging

# ---------------------------------------------------------Charts Class-----------------------------------------------------


class Charts(Base):

	def Categorical_missingness_Crosstab_Plot(independent, target):
	    
	    '''Plot cross tab'''
	    
	    missingness = independent.isnull()
	    cross_tab = pd.crosstab(target, missingness, normalize="columns", dropna=True).apply(lambda r: round(r,2), axis=1)

	    ax = cross_tab.plot(kind='bar', width=0.15, ylabel="Number Absorbed",color=["#003A5D","#A19958"]\
	   ,edgecolor="tab:grey",linewidth=1.5)

	    l = {"Not-Absorbed":"#003A5D", "Absorbed":"#A19958"}
	    labels = list(l.keys())
	    handles = [plt.Rectangle((5,5),10,10, color=l[label]) for label in labels]
	    plt.legend(handles, labels, fontsize=7, bbox_to_anchor=(1.13,1.17), loc="upper left", title="legend",shadow=True)

	    plt.title("Number Absorbed for each Gender", fontsize=9, pad=12)
	    plt.xlabel("Gender",fontsize=7.5)
	    plt.xticks(fontsize=7.5)
	    plt.ylabel('Number Absorbed', fontsize = 7.5)
	    plt.yticks(fontsize=7.5)
	    plt.rcParams["figure.figsize"] = (2.7,2.5)
	    plt.rcParams["legend.title_fontsize"] = 7

	    for pos in ["right", "top"]:
	        plt.gca().spines[pos].set_visible(False)
	        
	    for c in ax.containers:
	        ax.bar_label(c, label_type='edge', fontsize=7)
	     
	    return cross_tab


	def Categorical_missingness_Pivot_Plot(independent, target):
	      
	    '''Categorical Plot for greater than 2 categories'''
	    
	    missingness = independent.isnull()
	    df = pd.concat([missingness, target], axis=1) 
	    df_pivot = pd.pivot_table(df, index=independent.name, values=target.name, aggfunc=len, fill_value=0)\
	                                          #.apply(lambda x: x/float(x.sum()))

	    d = df_pivot.plot(kind="bar", width=0.1, color=["#003A5D","#A19958"], fontsize=7.5\
	                         , edgecolor="tab:grey",linewidth=1.5)

	    d.legend(title="legend", bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=6.5, shadow=True)
	    
	    plt.title("Race and Absorption for Gender", fontsize=7.5, pad=12)
	    plt.xlabel('Absorbed', fontsize=7)
	    plt.xticks(fontsize=7)
	    plt.ylabel('Number Absorbed', fontsize = 7)
	    plt.yticks(fontsize=7)
	    plt.xlabel(" ")
	    plt.rcParams["figure.figsize"] = (2.7,2.5)
	    plt.rcParams["legend.title_fontsize"] = 7

	    for pos in ["right", "top"]:
	        plt.gca().spines[pos].set_visible(False)


	    return df_pivot


	def Categorical_Crosstab_Plot(independent, target):
	    
	    ''' Plot cross tab '''

	    h = pd.crosstab(target,independent, normalize="columns")
	    bar = plt.bar(target, independent)
	    return plt.show(), h


	def Categorical_Pivot_Plot(independent, target):
	      
	    '''Categorical Plot for greater than 2 categories'''
	    
	    df = pd.concat([independent, target], axis=1) 
	    df_pivot = pd.pivot_table(df, index=independent.name, columns=target.name, aggfunc=len, fill_value=0)\
	                                                                       .apply(lambda x: x/float(x.sum()))
	    
	    return df_pivot.plot(kind="bar"), df_pivot


	def Scatter_Plot(independent, target):
	    
	    '''Scatter plot between numerical variables'''
	    
	    scatter = plt.scatter(target, independent)
	    return plt.show()   

	def Correlation_Plot(dataframe):
	    
	    '''Independent variables correlation plot'''

	    return dataframe.corr()

	def Point_Biserial_Plot(independent, target):
	    
	    sns.set_theme(style="ticks", color_codes = True)
	    data = pd.concat([independent, target], axis=1)
	    
	    return sns.catplot(x = independent, y = target, kind="box", data = data)   