
"""
    ========================================
     DATA CLUSTERING AND DIMENSION REDUCTION
     ========================================

     =======================
     K_Prototype Clustering
     =======================
 """

import missing_adhoc
from kmodes.kprototypes import KPrototypes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from collections import Counter

from class_base import Base

class ClusterCustomers(Base, object):

    def __init__(self, custom_rcParams, dataframe):

        self.custom_rcParams = plt.rcParams.update(custom_rcParams)

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in Base.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is class: {self.__class__.__name__}, and it provides functionalities for others"

    def k_prototypes_clustering(self, dataframe, cluster_no, categorical):

        values = list(np.arange(1,20,1))
        dataframe = missing_adhoc.df_loan_total_no_missing
        categorical_columns = list(np.arange(20, 28, 1))
        cluster_no = 5 
        columns = dataframe.columns.tolist()
        float_columns = []
        for i in values:
            float_columns.append(columns[i])        
        loan_norm = missing_adhoc.df_loan_total_no_missing.copy()
        scaler = preprocessing.MinMaxScaler()
        loan_norm[float_columns] = scaler.fit_transform(loan_norm[float_columns])
        kproto = KPrototypes(n_clusters=cluster_no, init="Cao")
        clusters = kproto.fit_predict(loan_norm, categorical=categorical_columns)
        labels = pd.DataFrame(clusters)
        labeled_df_loan = pd.concat([missing_adhoc.df_loan_total_no_missing, labels], axis=1)
        labeled_df_loan = labeled_df_loan.rename({0: "labels"}, axis=1)
         
        return labeled_df_loan

    def k_prototype_plot(self, independent, target, dataframe):
        
        self.fig, self.axs = plt.subplots(1,1)
        sns.swarmplot(x= self.independent, y= self.target, data=self.dataframe, hue="labels",
                           zorder=0, ax=self.axs)
        
        return self.fig   
    
    def frequency(self):

        data = ["Red", "Blue", "Red", "Green", "Blue", "Red", "Yellow", "Blue", "Green"]
        frequency_distribution = Counter(data)
        modes = [category for category, frequency in frequency_distribution.items()
                 if frequency == max(frequency_distribution.values())]

