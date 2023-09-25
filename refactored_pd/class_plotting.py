
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
import logging

from class_base import Base

# ---------------------------------------------------------Charts Class-----------------------------------------------------

class Charts(Base, objects):

    def __str__(self):
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in Charts.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def cat_missing_crosstab_plot(self):
        
        """ Plot cross tab """

        self.fig, self.axs = plt.subplots(1,1)
        missingness = independent.isnull()
        cross_tab = pd.crosstab(target, missingness, normalize="columns", dropna=True).apply(lambda r: round(r,2),
                                 axis=1)
        cross_tab.plot(kind='bar', width=0.15, ylabel="Number Absorbed",color=["#003A5D","#A19958"]\
       ,edgecolor="tab:grey",linewidth=1.5, ax=self.axs)
         
        return self.fig, cross_tab

    def cat_missing_pivot_plot(self):
          
        """ Categorical Plot for greater than 2 categories """

        self.fig, self.axs = plt.subplots(1,1)
        missingness = self.independent.isnull()
        df = pd.concat([missingness, self.target], axis=1) 
        df_pivot = pd.pivot_table(df, index=self.independent.name, values=self.target.name, aggfunc=len,
                                  fill_value=0) #.apply(lambda x: x/float(x.sum()))
        df_pivot.plot(kind="bar", width=0.1, color=["#003A5D","#A19958"], fontsize=7.5\
                             , edgecolor="tab:grey",linewidth=1.5, ax=self.axs)

        return self.fig, df_pivot

    def cat_crosstab_plot(self):
        
        """ Plot cross tab """

        self.fig, self.axs = plt.subplots(1,1)
        h = pd.crosstab(target,independent, normalize="columns")
        bars = self.axs.bar(self.independent, self.target, color='blue')
        key_color = {self.independent.name:"#003A5D", self.target.name:"#A19958"}

        return self.fig, h

    def cat_pivot_plot(self):
          
        """ Categorical Plot for greater than 2 categories """

        self.fig, self.axs = plt.subplots(1,1)
        df = pd.concat([self.independent, self.target], axis=1) 
        df_pivot = pd.pivot_table(df, index=self.independent.name, columns=self.target.name,
                                  aggfunc=len, fill_value=0).apply(lambda x: x/float(x.sum()))        
        df_pivot.plot(kind="bar", ax=self.axs)

        return self.fig, df_pivot

    def scatter_plot(self):
        
        """ Scatter plot between numerical variables """

        self.fig, self.axs = plt.subplots(1,1)
        self.axs.scatter(self.independent, self.target)

        return self.fig   

    def correlation_plot(self):
        
        """ Independent variables correlation plot """

        self.fig, self.axs = plt.subplots(1,1)
        self.dataframe.corr(ax=self.axs)

        return self.fig

    def point_biserial_plot(self):

        """ Point Biserial Plots """

        self.fig, self.axs = plt.subplots(1,1)
        sns.set_theme(style="ticks", color_codes = True)
        data = pd.concat([self.independent, self.target], axis=1)        
        sns.catplot(x = self.independent, y = self.target, kind="box", data = data, ax=self.axs) 

        return self.fig 

