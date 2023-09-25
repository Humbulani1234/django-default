
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from class_modelperf import ModelPeformance

class ClusterProbability(ModelPeformance, object):

    def __init__(self, custom_rcParams):

        self.custom_rcParams = plt.rcParams.update(custom_rcParams)

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ClusterProbability.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is class: {self.__class__.__name__}, with methods: {method_names}"
    
    def decile_method(self):
        
        self.fig, self.axs = plt.subplots(1,1)
        predict_probability = super(ModelPeformance, self).probability_prediction()
        np.random.seed(0)
        pd_values = np.random.uniform(0, 0.5, 1000)
        sorted_indices = np.argsort(pd_values)[::-1]
        sorted_pd_values = pd_values[sorted_indices]

        # Divide customers into deciles
        num_customers = len(sorted_pd_values)
        decile_size = num_customers // 10
        deciles = np.zeros(num_customers, dtype=int)

        for i in range(10):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size
            deciles[start_idx:end_idx] = i + 1

        # Print decile assignments for a few customers
        for customer_idx in range(10):
            print(f"Customer {customer_idx + 1}: PD = {sorted_pd_values[customer_idx]:.4f},\
                                                  Decile = {deciles[customer_idx]}")

    def _cluster_kmeans_initial(self):

        np.random.seed(0)
        pd_values = np.random.uniform(0, 0.5, 1000).reshape(-1, 1)  # Reshape for clustering

    def _elbow_max_cluster(self):

        # Determine the optimal number of clusters using the "elbow method"
        self.fig, self.axs = plt.subplots(1,1)
        inertia = []
        for n_clusters in range(1, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(pd_values)
            inertia.append(kmeans.inertia_)

        Plot the "elbow" curve to choose the number of clusters
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.grid(True)
        # plt.show()

    # Based on the elbow method, let's choose the number of clusters (e.g., 3)
    def kmeans_cluster(self):

        chosen_n_clusters = 3

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=chosen_n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(pd_values)
        # print(cluster_labels)
        # Assign cluster labels to each customer
        clustered_customers = np.hstack((pd_values, cluster_labels.reshape(-1, 1)))
        # print(clustered_customers)
        # print(type(clustered_customers))
        # Print a few samples of clustered customers
        for customer_idx in range(10):
            pd_value, cluster = clustered_customers[customer_idx]
            # print(f"Customer {customer_idx + 1}: PD = {pd_value:.4f}, Cluster = {cluster}")

    def kmeans_cluster_plot(self):

        # Assuming 'kmeans' is your k-means clustering model
        self.fig, self.axs = plt.subplots(1,1)
        cluster_centers = kmeans.cluster_centers_
        print(cluster_centers.shape)

        # Assuming 'labels' is the cluster assignment for each data point
        pd_values = pd.DataFrame(data=pd_values)
        pd_values['Cluster'] = cluster_labels  # Add cluster assignments to your data

        for cluster_id in pd_values['Cluster'].unique():
            cluster_data = pd_values[pd_values['Cluster'] == cluster_id]
            print(cluster_data)
            plt.hist(cluster_labels, bins=20, alpha=0.5, label=f'Cluster {cluster_id}')
        ax = sns.swarmplot(x= pd_values[0], y= cluster_labels, data=pd_values, hue="Cluster", zorder=0)
        ax.legend()

        plt.show()
