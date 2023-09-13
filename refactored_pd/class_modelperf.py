
"""
    =================
    MODEL ASSESSMENT
    =================
    
    And

    =======================
    Perfomance measurement
    =======================
    
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import sys

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

sys.path.append('/home/humbulani/django/django_ref/refactored_pd')

# --------------------------------------------------------Model Perfomance class----------------------------------------------------------

with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class ModelPerfomance(Base):

    def __init__(self, custom_rcParams, x_test, y_test, threshold):

        super().__init__(custom_rcParams)

        self.x_test = x_test
        self.y_test = y_test
        self.threshold = threshold
        self.predict_glm = loaded_model.predict(self.x_test)
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y_test, self.predict_glm)

    def roc_curve_analytics(self):
    
        """ Roc curve analytics and plot """

        self.fig, self.axs = plt.subplots(1,1)
        self.axs.plot(self.fpr, self.tpr)

        super().plotting("Roc Curve", "fpr", "tpr")

        return self.fig
   
    def optimal_threshold(self):

        self.optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal_thres = self.thresholds[self.optimal_idx]
        
        return self.optimal_thres

    def binary_prediction(self):
         
        """ Prediction Function @ maximal threshold """

        self.k = self.predict_glm.tolist()
        self.predict_binary = self.k.copy()

        for i in range(self.y_test.shape[0]):

            if self.predict_binary[i] < self.threshold:

                self.predict_binary[i] = 1               
        
            else: 

                self.predict_binary[i] = 0
            
            self.predict_binary = pd.Series(self.predict_binary)

        return self.predict_binary


    def confusion_matrix_plot(self):
        
        """ confusion matrix plot """

        self.fig, self.axs = plt.subplots(1,1) # find refactoring method
        predict_binary = self.binary_prediction()       
        conf_matrix = confusion_matrix(self.y_test, predict_binary, labels = [0, 1])
        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
        conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
        conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
        conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        return self.fig

    def probability_prediction(self):
         
        self._z = [round(i,10) for i in self.predict_glm.tolist()]
        prediction_prob = self._z.copy()

        return prediction_prob