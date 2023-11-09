
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay,
                             roc_auc_score,
                             roc_curve,
                             precision_score,
                             recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import set_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
from typing import Type
import seaborn as sns

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning_pd
from class_missing_values import ImputationCat

lg_logger = logging.getLogger("class_lgclassifier")
lg_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
lg_logger.addHandler(console_handler)
lg_logger.info("MODEL LOGISTIC REGRESSION IS INCLUDED")

class LogRegression(OneHotEncoding, object):

    """ This class encapsulates the process that must be followed to fit various Logistic regression models,
    it provides the functionalities or the steps into one comprehensive class, where it becomes a matter of just
    initialising the class and create an object with the required dataframes and just call the methods on it"""

    set_config(enable_metadata_routing=True)

    def __init__(self, 
                 custom_rcParams, 
                 df_nomiss_cat, 
                 type_, 
                 df_loan_float, 
                 target,
                 grid_search:Type[GridSearchCV], 
                 randomstate, 
                 onehot, 
                 threshold=None
    ):

        OneHotEncoding.__init__(self, custom_rcParams, df_nomiss_cat, type_, randomstate, onehot)  
        self.randomstate_lg = self.random_state_one
        self.threshold_lg = threshold
        self.df_loan_float_lg = df_loan_float
        self.target_lg = target
        self.x_train_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[0]
        self.y_train_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[1]
        self.x_val_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[2]
        self.y_val_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[3]
        self.x_test_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[4]
        self.y_test_lg = super(LogRegression, self).train_val_test(self.df_loan_float_lg, self.target_lg)[5]
        self.grid_search = grid_search

    def __str__(self):
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in LogisticRegression.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is class: {self.__class__.__name__}, and this is the public interface: {method_names}"

    def lg_grid_search(self):

        """ The following uses GridSearchCV to find the best l2 penalty regularization parameter
        GridSearchCV is being implemented as arm to the LogisticRegression class"""

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        gridsearch = self.grid_search(LogisticRegression(param_grid, cv=5, penalty="l2", random_state=self.randomstate))
        gridsearch.fit(self.x_train, self.y_train)
        best_c = gridsearch.best_params_['C']
        return best_c
    
    def lg_classification_fit(self):
        
        """ SGD Classification fit - Fit a base Logistic Classifier model"""
        
        lg_clf = LogisticRegression(random_state=self.randomstate_lg)
        lg_clf.fit(self.x_train_lg, self.y_train_lg)        
        return lg_clf

    def _lg_predict_prob_positive(self, x_test):

        """ Logistic Regression model prediction values - returns probabilities for the positive class """

        pred_prob = self.lg_classification_fit().predict_proba(x_test)[:,1]
        return pred_prob

    def lg_view_coef(self):

        """ Extract the intercept and the coefficients of the model - the predicted weights """

        coefficients = self.lg_classification_fit().coef_
        intercept = self.lg_classification_fit().intercept_
        return coefficients, intercept
    
    def lg_overfitting_test(self, x_train, y_train, x_test, y_test, *thresholds):

        train_precisions = []
        train_recalls = []
        test_precisions = []
        test_recalls = []
        for threshold in thresholds:

            y_prob_train = self._lg_predict_prob_positive(x_train)
            y_pred_train = (y_prob_train > threshold).astype(int)
            
            y_prob_test = self._lg_predict_prob_positive(x_test)
            y_pred_test = (y_prob_test > threshold).astype(int)

            train_precision = precision_score(y_train, y_pred_train)
            train_recall = recall_score(y_train, y_pred_train)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)

            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)

        self.fig, (self.axs1, self.axs2)  = plt.subplots(1, 2)
        self.axs1.plot(thresholds, train_precisions, label = "Training Precision")
        self.axs1.plot(thresholds, test_precisions, label='Testing Precision')
        self.axs1.set_label('Threshold')
        self.axs1.set_label('Precision')
        self.axs1.legend()
        self.axs2.plot(thresholds, train_recalls, label = "Training Recalls")
        self.axs2.plot(thresholds, test_recalls, label='Testing Recalls')
        self.axs2.set_label('Threshold')
        self.axs2.set_label('Precision')
        self.axs2.legend()
        return self.fig, train_precisions, train_recalls

    def lg_feature_importance(self):

        """ Determination of feature importance """

        feature_importances = self.lg_classification_fit().coef_[0]
        feature_names = self.x_train_lg.columns
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        self.fig, self.axs = plt.subplots(1,1)
        self.axs.barh(*zip(*sorted_feature_importance))
        self.axs.set_label('Feature Importance')
        self.axs.set_label('Feature Name')
        self.axs.title('Feature Importance in Decision Tree')
        return self.fig

    def lg_sample_class_pred(self, sample, input_sample):
        
        """ Classification prediction - It classifies a customer based on a threshold: whether they are 
        postive or negative """
        
        pred_prob = self._lg_predict_prob_postive(input_sample).round(10)
        if self.threshold_lg is None:
            raise ValueError("Provide the threshold value in the class constructor")
        else:
            if pred_prob[sample] >= self.threshold: # assign threshold manually
                return 1
            else:
                return 0

    def lg_sample_prob_pred(self, sample, input_sample):
        
        """ Probability prediction - It returns the probability that is greater than the threshold:
        The postive class or else it returns the negative class probability """
        
        pred_prob = self.lg_classification_fit().predict_proba(input_sample).round(10)
        if self.threshold_lg is None:
            raise ValueError("Provide the threshold value in the class constructor")
        if pred_prob[sample][1] >= self.threshold_lg:
            return pred_prob[sample][1]
        else:
            return pred_prob[sample][0]

    def lg_optimal_threshold(self, x_test, y_test):

        fpr, tpr, thresholds = roc_curve(y_test, self._lg_predict_prob_positive(x_test))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thres = thresholds[optimal_idx]
        return optimal_thres

    def lg_binary_prediction(self, x_test, threshold=None):
         
        """ Binary prediction function threshold or user supplied threshold"""

        if  threshold is None:
            print("running with the default 0.5 threshold")
            pred_bin = self.lg_classification_fit().predict(x_test)
            return pred_bin
        else:            
            predict_binary = self._lg_predict_prob_positive(x_test)        
            for i in range(x_test.shape[0]):
                if predict_binary[i] >= threshold:
                   predict_binary[i] = 1                   
                else: 
                    predict_binary[i] = 0            
            return  predict_binary
   
    def lg_perf_analytics(self, x_test, y_test, threshold=None):
    
        """ Roc curve analytics and plot - Rocs points represent confusion matrices at varying 
        thresholds, default model threshold is 0.5 """

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

        if  threshold is None:
            print("running with the default 0.5 threshold")
            plt.close('all')
            self.fig, self.axs = plt.subplots(1,1)
            y_bin= self.lg_binary_prediction(x_test)
            optimal_thres = self.lg_optimal_threshold(x_test, y_test)
            accuracy = accuracy_score(y_test, y_bin)
            f1 = f1_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_bin)
            threshold = 0.5
            data = pd.DataFrame({
            "Metric": ["threshold", "accuracy", "f1", "auc"],
            "Value": [threshold, accuracy, f1, auc]
            }) 
            generate(data, self.axs, "LG")
            return self.fig, optimal_thres, accuracy, f1, auc
        else:
            plt.close('all')
            self.fig, self.axs = plt.subplots(1,1)
            y_bin = self.lg_binary_prediction(x_test, threshold)
            accuracy = accuracy_score(y_test, y_bin)
            f1 = f1_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_bin)
            threshold = threshold
            data = pd.DataFrame({
            "Metric": ["threshold", "accuracy", "f1", "auc"],
            "Value": [threshold, accuracy, f1, auc]
            }) 
            generate(data, self.axs, "LG")
            return self.fig, threshold, accuracy, f1, auc

    def lg_confusion_matrix_plot(self, x_test, y_test, threshold):

        """ Base tree Confusion matrix """

        def conf_plot():
            conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
            conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
            conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
            conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        self.fig, self.axs = plt.subplots(1,1)  
        pred_bin = pd.Series(self.lg_binary_prediction(x_test, threshold))   
        conf_matrix = confusion_matrix(y_test, pred_bin)
        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_plot()
        return self.fig, conf_matrix
