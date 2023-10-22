
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

from class_modelperf import ModelPerfomance
from class_base import Base

diagnostics_logger = logging.getLogger("class_clustering_pd")
diagnostics_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
diagnostics_logger.addHandler(console_handler)
diagnostics_logger.info("PROBABILITY CLUSTERING ACCORDING TO RISK")

class ClusterProbability(ModelPerfomance, Base, object):

    def __init__(self, custom_rcParams, x_test, y_test, threshold):
        super(ClusterProbability,self).__init__(custom_rcParams, x_test, y_test, threshold)
        super(ModelPerfomance,self).__init__(custom_rcParams)
        self.pd_values = super().probability_prediction()

    def __str__(self):
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ClusterProbability.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is class: {self.__class__.__name__}, with methods: {method_names}"
    
    def decile_method(self):        
        self.fig, self.axs = plt.subplots(1,1)
        predict_probability = self.pd_values
        sorted_indices = np.argsort(self.pd_values)[::-1]
        sorted_pd_values = self.pd_values[sorted_indices]
        num_customers = len(sorted_pd_values)
        decile_size = num_customers # 10
        deciles = np.zeros(num_customers, dtype=int)
        for i in range(10):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size
            deciles[start_idx:end_idx] = i + 1
        for customer_idx in range(10):
            print(f"Customer {customer_idx + 1}: PD = {sorted_pd_values[customer_idx]:.4f},\
                                                  Decile = {deciles[customer_idx]}")

    def _elbow_max_cluster(self):

        pd_values = np.array(self.pd_values).reshape(-1, 1)
        inertia = []
        for n_clusters in range(1, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(pd_values)
            inertia.append(kmeans.inertia_)
        self.fig, self.axs = plt.subplots(1,1)
        self.axs.plot(range(1, 11), inertia, marker='o', linestyle='--')
        super(ModelPerfomance,self)._plotting("Elbow method", "cluster no", "Inertia")

        return self.fig

    def kmeans_cluster(self):

        chosen_n_clusters = 3
        kmeans = KMeans(n_clusters=chosen_n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(np.array(self.pd_values).reshape(-1, 1))
        clustered_customers = np.hstack((self.pd_values, cluster_labels))

        return kmeans, cluster_labels


    def kmeans_cluster_plot(self):

        self.fig, self.axs = plt.subplots(1, 1)
        cluster_centers = self.kmeans_cluster()[0].cluster_centers_
        pd_values_df = pd.DataFrame(data=self.pd_values)
        pd_values_df['Cluster'] = self.kmeans_cluster()[1]
        key_color = {"Cluster 0": "lightgreen", "Cluster 1": "orange", "Cluster 2": "lightblue"}
        sns.swarmplot(x= self.kmeans_cluster()[1], y= pd_values_df[0], data=pd_values_df, hue="Cluster",
                      ax=self.axs, marker="o", palette="Set2")
        super(ModelPerfomance,self)._plot_legend(key_color)
        super(ModelPerfomance,self)._plotting("Customer Risk Clustering", "Cluster", "Probability")

        return self.fig