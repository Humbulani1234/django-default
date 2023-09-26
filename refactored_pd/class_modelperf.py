
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
import re
import logging

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

diagnostics_logger = logging.getLogger("class_modelperf")
diagnostics_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
diagnostics_logger.addHandler(console_handler)
diagnostics_logger.info("MODEL PERFOMANCE ARE INCLUDED")

sys.path.append('/home/humbulani/django/django_ref/refactored_pd')
with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class ModelPerfomance(Base, object):

    def __init__(self, custom_rcParams, x_test, y_test, threshold):

        super(ModelPerfomance, self).__init__(custom_rcParams)
        self.x_test = x_test
        self.y_test = y_test
        self.threshold = threshold

    def __str__(self):
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ModelPerfomance.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def _predict_glm(self):

        predict_glm = loaded_model.predict(self.x_test)
        return predict_glm

    def roc_curve_analytics(self):
    
        """ Roc curve analytics and plot """

        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self._predict_glm())
        self.fig, self.axs = plt.subplots(1,1)
        self.axs.plot(fpr, tpr)
        super(ModelPerfomance, self)._plotting("Roc Curve", "fpr", "tpr")

        return self.fig
   
    def optimal_threshold(self):

        optimal_idx = np.argmax(self.tpr - self.fpr)
        optimal_thres = self.thresholds[optimal_idx]
        
        return optimal_thres

    def binary_prediction(self):
         
        """ Prediction Function @ maximal threshold """

        predict_binary = self._predict_glm().tolist()        
        for i in range(self.y_test.shape[0]):
            if predict_binary[i] < self.threshold:
               predict_binary[i] = 1                   
            else: 
                predict_binary[i] = 0            
                predict_binary = pd.Series(predict_binary)

        return  predict_binary

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
         
        prediction_prob = [round(i,10) for i in self._predict_glm().tolist()]
        
        return prediction_prob

