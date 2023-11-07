
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
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import logging
from scipy.stats import chi2_contingency
import seaborn as sns
import statsmodels.api as sm
from math import *
import re

from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from class_traintest import OneHotEncoding

eda_logger = logging.getLogger("class_eda")
eda_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
eda_logger.addHandler(console_handler)
eda_logger.info("MODEL PERFOMANCE ARE INCLUDED")

# ---------------------------------------------------------EDA Class-----------------------------------------------------

class ExploratoryDataAnalysis(OneHotEncoding, object):

    """ This class provides features vs target plotting and hypothesis tests to determine if the features 
    are relevant to the model"""

    def __init__(self, custom_rcParams, df_nomiss_cat: pd.DataFrame, type_, df_loan_float, target):        
        super(ExploratoryDataAnalysis, self).__init__(custom_rcParams, df_nomiss_cat, type_)
        self.df_loan_float_eda = df_loan_float
        self.target_eda = target 
        self.x_eda = super(ExploratoryDataAnalysis, self).create_xy_frames(self.df_loan_float_eda, self.target_glm)[0]
        self.y_eda = super(ExploratoryDataAnalysis, self).create_xy_frames(self.df_loan_float_eda, self.target_glm)[1]

    def __str__(self):        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ExploratoryDataAnalysis.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def sample_imbalance(self, target):
    
        self.fig, self.axs = plt.subplots(1,1)        
        self.axs.hist(target, weights = np.ones(len(target))/len(target))
        # super().plotting("Normality Test", "x", "y")
        return self.fig

    def hist_float(self, target):
    
        self.fig, self.axs = plt.subplots(1,1)        
        self.axs.hist(target, bins=20, density=True, alpha=0.5, color="b", edgecolor="k", label="Histogram")
        sns.kdeplot(target, color="r", label="Density Curve")
        plt.legend()
        return self.fig

    def cat_missing_crosstab_plot(self, independent, target):
        
        """ Plot cross tab for features with missing values """

        self.fig, self.axs = plt.subplots(1,1)
        missingness = independent.isnull()
        cross_tab = pd.crosstab(missingness, target)
        plot = cross_tab.plot(kind='bar', width=0.15, ylabel=target.name, color=["#003A5D","#A19958"],
                       edgecolor="tab:grey", linewidth=1.5, ax=self.axs)
        bars = plot.containers[0] # change here to control the bars
        for bar in bars:
            p = str(bar.get_height())
            yval = bar.get_height()
            self.axs.text(bar.get_x(), yval + 3, p,fontsize=9)            
        chi_val, p_val, dof, expected = chi2_contingency(cross_tab)         
        return self.fig, cross_tab

    def cat_missing_pivot_plot(self, independent, target):
          
        """ Categorical Plot for greater than 2 categories - with features that have missing values """

        self.fig, self.axs = plt.subplots(1,1)
        missingness = independent.isnull()
        df = pd.concat([missingness, target], axis=1) 
        df_pivot = pd.pivot_table(df, index=independent.name, values=target.name, aggfunc=len,
                                  fill_value=0) #.apply(lambda x: x/float(x.sum()))
        df_pivot.plot(kind="bar", width=0.1, color=["#003A5D","#A19958"], fontsize=7.5,
                      edgecolor="tab:grey",linewidth=1.5, ax=self.axs)
        chi_val, p_val, dof, expected = chi2_contingency(df_pivot)

        return self.fig, df_pivot

    def cat_crosstab_plot(self, independent, target):
        
        """ Plot cross tab table and conduct chi-square testing - for features without missing values:

        H0:
        H1:

        """

        self.fig, self.axs = plt.subplots(1,1)
        crosstab = pd.crosstab(independent, target)
        crosstab.plot(kind="bar", xlabel=independent.name, ylabel=target.name,
        ax=self.axs, color = {"#003A5D", "#A19958"})
        chi_val, p_val, dof, expected = chi2_contingency(crosstab)
        return self.fig, crosstab, chi_val, p_val, dof, expected

    def cat_pivot_plot(self, independent, target):
          
        """ Categorical Plot using Pivot Tables - provide more flexibility, not only counts/frequencies
        for features without missing values """

        self.fig, self.axs = plt.subplots(1,1)
        df = pd.concat([independent, target], axis=1) 
        df_pivot = pd.pivot_table(df, index=independent.name, columns=target.name,
                                  aggfunc=len, fill_value=0).apply(lambda x: x/float(x.sum()))        
        df_pivot.plot(kind="bar", ax=self.axs)
        chi_val, p_val, dof, expected = chi2_contingency(df_pivot)
        return self.fig, df_pivot, p_val

    def scatter_plot(self, independent, target):
        
        """ Scatter plot between numerical variables and it can still be used between numerical and
         categorical(target)"""

        self.fig, self.axs = plt.subplots(1,1)
        self.axs.scatter(independent, target)
        return self.fig   

    def correlation_plot(self, dataframe):
        
        """ Independent variables correlation plot - assessment of multicollinearity"""

        self.fig, self.axs = plt.subplots(1,1)
        corr = dataframe.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=self.axs)
        return self.fig, corr

    def pearson_corr_test(self, independent, target):
        
        """ Pearson correlation test function """
        
        pearson_coef, p_value = pearsonr(independent, target)
        return pearson_coef, p_value

    def _corr_value_greater(self, dataframe, corr_threshold):

        """ Multicollinearity investigation, a function that returns correlation values greater than some
        supplied threshold """
        
        dataframe_corr = pd.DataFrame(data=dataframe.corr())
        g = []
        for i in range(dataframe_corr.shape[0]):
            for j in range(dataframe_corr.shape[0]):                
                if (dataframe_corr.iloc[i,j]>corr_threshold) or (dataframe_corr.iloc[i,j]<-corr_threshold):
                    g.append(dataframe_corr.iloc[i,j])   
        return g 

    def _get_indexes(self, dataframe, value):

        """ A function to retrieve row and column indexes when the value is equal to the supplied input"""
        
        list_of_pos = []
        dataframe_corr = pd.DataFrame(data=dataframe.corr())
        result = dataframe_corr.isin([value])
        series_obj = result.any()
        column_names = list(series_obj[series_obj == True].index)    
        for col in column_names:    
            rows = list(result[col][result[col] == True].index)            
            for row in rows:
                list_of_pos.append((row,col))        
        return list_of_pos

    def get_var_corr_greater(self, dataframe, corr_thershold):
           
        """ Get the tuple of features (different features) all with correlation greater than the specified
        correlation threshold """

        g_1 = self._corr_value_greater(dataframe, corr_thershold)
        tuple_corr = []
        for i in g_1:
            t = self._get_indexes(dataframe, i)
            tuple_corr.append([item for item in t if item[0]!=item[1]])    
        return tuple_corr

    def vif_value(self, dataframe, target):
        
        '''Calculate variance inflation factor'''

        ols = sm.regression.linear_model.OLS(target, dataframe.
                                                      drop(labels=[self.target.name]), axis=1)
        res_ols = ols.fit()              
        VIF = 1/(1-res_ols.rsquared_adj**2)        
        return VIF

    def point_biserial_plot(self, independent, target):

        """ Point Biserial Plots - plot between numerical and categorical variabales
         (swaped target and independent vars for plotting)"""

        self.fig, self.axs = plt.subplots(1,1)
        sns.set_theme(style="ticks", color_codes = True)
        data = pd.concat([target, independent], axis=1)        
        sns.boxplot(x = target.name, y = independent.name, data = data, ax=self.axs) 
        return self.fig 

    def point_biserial(self):
        
        """ Point Biserial Test for Binary vs Numerical variables """
        
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

    def feature_selection(self):

        """ Train each feature individually and then decide which features have a high AUC of the ROC
        Curve or we can iteratively add a single feature in the set to perform greedy selection and 
        select the feature with result in the least ERM predictor """
        for feature in self.x_train.column:

            lg_clf = LogisticRegression(random_state=self.randomstate)
            lg_clf.fit(self.x_train, self.y_train) 

if __name__ == "__main__":

    file_path = "./KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)
    custom_rcParams = {"figure.figsize": (9, 8), "axes.labelsize": 12}
    miss = ImputationCat(df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()
    print(df_loan_categorical.columns)
    print(df_loan_float.columns) 
    c = ExploratoryDataAnalysis(custom_rcParams, imputer_cat, "statistics", df_loan_float, df_loan_float["GB"])
    c.hist_float(df_loan_float['AGE'])
    # print(c._get_indexes(df_loan_float, 1))
    # print(c.get_var_corr_greater(df_loan_float, 0.7))
    # print(c.correlation_plot(df_loan_float)[0])
    # print(c.sample_imbalance())
    # print(c.scatter_plot())
    # print(c.point_biserial_plot())
    print(c.cat_crosstab_plot()[1])
    plt.show()