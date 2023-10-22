
import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm
from matplotlib import pyplot as plt
from typing import Type
from sklearn.model_selection import GridSearchCV
import seaborn as sns

import pd_download
from class_traintest import OneHotEncoding
from class_base import Base
from class_missing_values import ImputationCat
from class_modelperf import ModelPerfomance
from class_decision_tree import DecisionTree
from class_diagnostics import (ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
                               PartialPlots, LevStudQuaRes, CooksDisQuantRes, QuantileResiduals)
from class_lgclassifier import LogRegression

class ModelComparison(ModelPerfomance, DecisionTree, LogRegression, object):

    """ This class gathers all the model classes and performs model comparison 
    for this class to be more useful it has to be made independent of the models it is comparing"""

    def __init__(self, custom_rcParams, df_nomiss_cat, type_1, type_2,
                  df_loan_float, target, randomstate, grid_search:Type[GridSearchCV], threshold=None):

        ModelPerfomance.__init__(self, custom_rcParams, df_nomiss_cat, type_1,
                         df_loan_float, target, randomstate, threshold)
        DecisionTree.__init__(self, custom_rcParams, df_nomiss_cat, type_2,
                         df_loan_float, target, randomstate, threshold)
        LogRegression.__init__(self, custom_rcParams, df_nomiss_cat, type_2,
                         df_loan_float, target, randomstate, grid_search, threshold)

    def cmp_performance_metrics(self, ccpalpha, threshold_1, threshold_2, threshold=None):

        """ Calculation of recall, precision, accuracy and F1 score based on user supplied threshold
        For Postive Clss -(1) """

        """ Run analytics for GLM Logistic Regression """

        self.x_val_glm_n = sm.add_constant(self.x_val_glm.values, has_constant='add')
        threshold_glm, accuracy_glm, f1_glm, auc_glm = super(ModelComparison, self).glm_perf_analytics(self.x_val_glm_n, 
                                                                                                       self.y_val_glm)[1],\
                                           super(ModelComparison, self).glm_perf_analytics(self.x_val_glm_n, self.y_val_glm)[2],\
                                           super(ModelComparison, self).glm_perf_analytics(self.x_val_glm_n, self.y_val_glm)[3],\
                                           super(ModelComparison, self).glm_perf_analytics(self.x_val_glm_n, self.y_val_glm)[4],\

        """ Run analytics for DecisionTree """

        threshold_dt, accuracy_dt, f1_dt, auc_dt, = super(ModelPerfomance, self).dt_pruned_perf_analytics(ccpalpha, threshold_1,
                                                                                         threshold_2, self.x_val_dt, 
                                                                                         self.y_val_dt, threshold)[1],\
                                            super(ModelPerfomance, self).dt_pruned_perf_analytics(ccpalpha, threshold_1,
                                                                                         threshold_2, self.x_val_dt, 
                                                                                         self.y_val_dt, threshold)[2],\
                                            super(ModelPerfomance, self).dt_pruned_perf_analytics(ccpalpha, threshold_1,
                                                                                         threshold_2, self.x_val_dt, 
                                                                                         self.y_val_dt, threshold)[3],\
                                            super(ModelPerfomance, self).dt_pruned_perf_analytics(ccpalpha, threshold_1,
                                                                                         threshold_2, self.x_val_dt, 
                                                                                         self.y_val_dt, threshold)[4],\

        """ Run analtics for LogRegression """

        threshold_lg, accuracy_lg, f1_lg, auc_lg = super(DecisionTree, self).lg_perf_analytics(self.x_val_lg, self.y_val_lg,
                                                                                               threshold)[1],\
                                           super(DecisionTree, self).lg_perf_analytics(self.x_val_lg, self.y_val_lg, threshold)[2],\
                                           super(DecisionTree, self).lg_perf_analytics(self.x_val_lg, self.y_val_lg, threshold)[3],\
                                           super(DecisionTree, self).lg_perf_analytics(self.x_val_lg, self.y_val_lg, threshold)[4]
        plt.close('all')
        if __name__ == "__main__":
            print(f"optimal_thres for GLM Regression is: {threshold_glm}")
            print(f"accuracy for GLM Regression is: {accuracy_glm}")
            print(f"F1 Score for GLM Regression is: {f1_glm}")
            print(f"AUC for GLM Regression is: {auc_glm}")

            print(f"optimal_thres for DT is: {threshold_dt}")
            print(f"accuracy for DT is: {accuracy_dt}")
            print(f"F1 Score for DT is: {f1_dt}")
            print(f"AUC for DT is: {auc_dt}")

            print(f"optimal_thres for LG Regression is: {threshold_lg}")
            print(f"accuracy for LG Regression is: {accuracy_lg}")
            print(f"F1 Score for LG Regression is: {f1_lg}")
            print(f"AUC for LG Regression is: {auc_lg}")

        self.fig, (self.axs1, self.axs2, self.axs3)  = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.4) 
        data1 = pd.DataFrame({
        "Metric": ["thres", "acc", "f1", "auc"],
        "Value": [threshold_glm, accuracy_glm, f1_glm, auc_glm]
        })
        data2 = pd.DataFrame({
        "Metric": ["thres", "acc", "f1", "auc"],
        "Value": [threshold_dt, accuracy_dt, f1_dt, auc_dt]
        }) 
        data3 = pd.DataFrame({
        "Metric": ["thres", "acc", "f1", "auc"],
        "Value": [threshold_lg, accuracy_lg, f1_lg, auc_lg]
        })  

        def generate(data, axs, title_):
            sns.set_theme(style="ticks", color_codes=True)    
            sns.barplot(x="Metric", y="Value", data=data, ax=axs)
            axs.spines["top"].set_visible(False)  
            axs.spines["right"].set_visible(False) 
            axs.set_title(title_)
            for p in axs.patches:
                axs.annotate(f"{p.get_height().round(1)}", (p.get_x().round(2)+p.get_width().round(2)/2.,
                                   p.get_height().round(2)), ha="center",va="center",fontsize=12, color="black",
                                   xytext=(0,5),textcoords="offset points")  

        generate(data1, self.axs1, "GLM")
        generate(data2, self.axs2, "DT")
        generate(data3, self.axs3, "LG")      
        return self.fig

    def cmp_overfitting(self, ccpalpha, threshold_1, threshold_2, *thresholds):
    
        """ Roc curve analytics and plot - Rocs points represent confusion matrices at varying 
        thresholds, default model threshold is 0.5 """

        """ Run overffiting for GLM Logistic Regression """
        
        self.x_val_glm_c = sm.add_constant(self.x_val_glm.values, has_constant='add')
        self.x_test_glm_c = sm.add_constant(self.x_test_glm.values, has_constant='add')
        val_precisions_glm, val_recalls_glm = super(ModelComparison, self).glm_overfitting_test(self.x_val_glm_c, self.y_val_glm,
                                                                                        self.x_test_glm_c, self.y_test_glm,
                                                                                         *thresholds)[1],\
                                            super(ModelComparison, self).glm_overfitting_test(self.x_val_glm, self.y_val_glm,
                                                                                        self.x_test_glm, self.y_test_glm, 
                                                                                        *thresholds)[2]

        """ Run overfiting for Decision Trees """

        val_precisions_dt, val_recalls_dt = super(ModelPerfomance, self).dt_pruned_overfitting(ccpalpha, threshold_1, 
                                                                           threshold_2, self.x_val_dt, self.y_val_dt,
                                                                           self.x_test_dt, self.y_test_dt, *thresholds)[1],\
                                            super(ModelPerfomance, self).dt_pruned_overfitting(ccpalpha, threshold_1, 
                                                                    threshold_2, self.x_val_dt, self.y_val_dt, 
                                                                    self.x_test_dt, self.y_test_dt, *thresholds)[2]

        """ Run overfitting for LogisticRegression """

        val_precisions_lg, val_recalls_lg = super(DecisionTree, self).lg_overfitting_test(self.x_train_lg, self.y_train_lg,
                                                                            self.x_test_lg, self.y_test_lg, *thresholds)[1],\
                                            super(DecisionTree, self).lg_overfitting_test(self.x_train_lg, self.y_train_lg,
                                                                            self.x_test_lg, self.y_test_lg, *thresholds)[2]
        plt.close('all')
        self.fig, (self.axs1, self.axs2)  = plt.subplots(1, 2)
        self.axs1.plot(thresholds, val_precisions_glm, label = "Validation_GLM Precision")
        self.axs1.plot(thresholds, val_precisions_dt, label='Validation_DT Precision')
        self.axs1.plot(thresholds, val_precisions_lg, label='Validation_LG Precision')
        self.axs1.set_label('Threshold')
        self.axs1.set_label('Precision')
        self.axs1.legend()
        self.axs2.plot(thresholds, val_recalls_glm, label = "Testing_GLM Recalls")
        self.axs2.plot(thresholds, val_recalls_dt, label='Testing_DT Recalls')
        self.axs2.plot(thresholds, val_recalls_lg, label='Testing_LG Recalls')
        self.axs2.set_label('Threshold')
        self.axs2.set_label('Precision')
        self.axs2.legend()
        return self.fig

    def cmp_confusion_matrix_plot(self, ccpalpha, threshold_1, threshold_2, threshold=None):

        """ Confusion matrix for GLM - Logistic Regression """

        self.x_val_glm_n = sm.add_constant(self.x_val_glm.values, has_constant='add')
        glm_conf_matrix = super(ModelComparison, self).confusion_matrix_plot(self.x_val_glm_n, self.y_val_glm)[1]
    
        """ Confusion Matrix for Decision Trees """

        dt_conf_matrix = super(ModelPerfomance, self).dt_pruned_conf_matrix(ccpalpha, threshold_1, threshold_2,
                                                                   self.x_val_dt, self.y_val_dt, threshold)[1]

        """ Confusion Matrix for LogisticRegression """

        lg_conf_matrix = super(DecisionTree, self).lg_confusion_matrix_plot(self.x_val_lg, self.y_val_lg, threshold)[1]
        plt.close('all')
        self.fig, (self.axs1, self.axs2, self.axs3)  = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.4) 

        def generate(axs, conf, title_):
            im = axs.matshow(conf, cmap='viridis')
            axs.set_xticklabels(['', 'No Default', 'Yes Default'])
            axs.xaxis.set_ticks_position('bottom')
            axs.set_yticklabels(['', 'No Default', 'Yes Default'], rotation=90, va='center')
            axs.set_xlabel('Predicted')
            axs.set_ylabel('Actual')
            axs.set_title(title_, pad=15)
            for i in range(2):
                for j in range(2):
                    axs.text(j, i, str(conf[i,j]), ha="center")
            return im

        generate(self.axs1, glm_conf_matrix, 'Confusion Matrix GLM')
        generate(self.axs2, dt_conf_matrix, 'Confusion Matrix DT')
        generate(self.axs3, lg_conf_matrix, 'Confusion Matrix LG')          
        return self.fig

