
"""
    =============================
    GLM LOGISTIC MODEL ASSESSMENT
    =============================
    And
    =======================
    Perfomance measurement
    =======================    
"""
from sklearn.metrics import (accuracy_score,
                            f1_score, 
                            mean_squared_error,
                            r2_score,
                            roc_curve,
                            precision_score,
                            recall_score,
                            confusion_matrix,
                            ConfusionMatrixDisplay,
                            roc_auc_score)
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import sys
import re
import sympy as sp
import math
import scipy
import logging
import seaborn as sns

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

modelperf_logger = logging.getLogger("class_modelperf")
modelperf_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
modelperf_logger.addHandler(console_handler)
modelperf_logger.info("MODEL PERFOMANCE ARE INCLUDED")

sys.path.append('/home/humbulani/django/django_ref/refactored_pd')
with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class ModelPerfomance(OneHotEncoding, object):

    """ This Class deals with Model Perfomanace, including addressing issues such as: 

        1) Model Overfitting through calculation of Precision and Recall statistics for Trainning
           Validation and Testing datasets;
        2) ROC Curve analytics;
        3) Plotting fo Confusion for maximal value of different thresholds (maximal threshold)
        4) Feature significance through p-value Hypothesis testing - read from the model output
        5) Regression (Logistic) equation - expressed algebraically 

        This object is created in such a way that it has to behave accordingly to the supplied
        dataframes during initialisation process, meaning the methods (mostly) will not require dataframes user
        inputs whencalled on the object
    """

    feature_mapping = {

        "const": "const",
        "x1":"CHILDREN", "x2": "PERS_H", "x3": "AGE", "x4": "TMADD", "x5": "TMJOB1",
        "x6": "TEL", "x7": "NMBLOAN", "x8": "FINLOAN", "x9": "INCOME", 
        "x10": "EC_CARD", "x11": "INC", "x12": "INC1", "x13": "BUREAU", 
        "x14": "LOCATION", "x15": "LOANS", "x16": "REGN", "x17": "DIV", "x18": "CASH", 
        "x19": "H", "x20": "E", "x21": "G", "x22": "T", "x23": "U", "x24": "V", "x25": "Cars",
        "x26": "Dept_Store_Mail", "x27": "Furniture_Carpet", "x28": "Leisure", "x29": "OT",
        "x30": "Lease", "x31": "German", "x32": "Greek", "x33": "Italian", "x34": "Other_European",
        "x35": "RS", "x36": "Spanish_Portugue", "x37": "Turkish", "x38": "Chemical_Industr",
        "x39": "Civil_Service_M", "x40": "Food_Building_Ca", "x41": "Military_Service",
        "x42": "Others", "x43": "Pensioner", "x44": "Sea_Vojage_Gast", "x45": "Self_employed_pe",
        "x46": "Car", "x47": "Car_and_Motor_bi", "x48": "American_Express", "x49": "Cheque_card",
        "x50": "Mastercard_Euroc", "x51": "Other_credit_car", "x52": "VISA_Others", "x53": "VISA_mybank"
    }

    def __init__(self, custom_rcParams, df_nomiss_cat, type_, df_loan_float, target, randomstate, threshold=None):

        """ Object initialisation attributes """

        OneHotEncoding.__init__(self, custom_rcParams, df_nomiss_cat, type_)
        self.df_loan_float_glm = df_loan_float
        self.target_glm = target 
        self.x_train_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[0]
        self.y_train_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[1]
        self.x_val_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[2]
        self.y_val_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[3]
        self.x_test_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[4]
        self.y_test_glm = super(ModelPerfomance, self).train_val_test(self.df_loan_float_glm, self.target_glm)[5]  
        self.threshold_glm = threshold
        self.randomstate_glm = randomstate

    def __str__(self):

        """ A print of the object """

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ModelPerfomance.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"""This is Class {self.__class__.__name__} with methods {method_names} and before any feature analysis 
          my regression equation is given by:
          
          {self.regression_equation()}"""

    @classmethod
    def glm_rename_feature_names(cls):

        """ Renaming regression features to correspond to model features """

        coef_table = loaded_model[0].tables[1]
        coef_df = pd.DataFrame(coef_table.data, columns=coef_table.data[0])
        coef_df = coef_df[1:].rename(columns={'': 'feature'})
        coef_df['feature'] = coef_df['feature'].map(cls.feature_mapping)
        return coef_df

    def glm_extract_sig_features(self):

        """ Extracting significant features based on the p-values, conducted through Wald Test
        presented in regression output"""

        coef_df = self.glm_rename_feature_names()
        coef_df['P>|z|'] = coef_df['P>|z|'].astype(float)
        significant_features = (coef_df['P>|z|'] < 0.05)
        sig_df = coef_df[significant_features]
        return sig_df

    def glm_reg_equ(self):

        """ Expressing a regression equation to displayed during a call to the __repr__/print method"""
        
        coef_df = self.glm_rename_feature_names()
        intercept = coef_df['coef'].astype(float).to_list()[0]
        coefficients = coef_df['coef'].astype(float).to_list()[1:]
        feature_names = coef_df['feature'].to_list()[1:]
        equation = "log odds = {:.4f} + ".format(intercept)
        for i, coef in enumerate(coefficients):
            equation += "{:.4f} * {} + ".format(coef, feature_names[i])
        equation = equation[:-3]
        return equation

    def glm_reg_algebraic_equ(self, feature_values):

        """ Using the regression equation by converting it to an algebraic expression first to conduct calculations,
        like log odds effects on the predictor """

        equation, coef_df = self.glm_reg_equ(), self.glm_rename_feature_names() 
        feature_names = coef_df['feature'].to_list()[1:]
        tuple_str = (str(tuple(feature_names))[1:-1]).replace("'", "")
        input_str = tuple_str.replace(",", "")
        equation = sp.smpify(equation)
        final_str = sp.symbols("input_str")
        log_odds = sp.lambdify(tuple(final_str), equation, 'numpy')
        return log_odds(feature_values)

    def glm_revised_log_model(self):

        """ After extracting significant features, fit the model again"""

        new_features = self.glm_extract_sig_features()["feature"].values
        clean_feature= new_features[-1].replace("_", "/") # Clean the last feature
        new_features[-1] = clean_feature
        new_x_train = self.x_train_glm[new_features]
        return glm_binomial_fit(self.y_train_glm, new_x_train)

    def _glm_predict_pos_prob(self, x_test):

        """ GLM: logistic regression model prediction values - returns probabilities """

        predict_glm = loaded_model[1].predict(x_test)
        return predict_glm

    def glm_sample_class_pred(self, sample, input_sample):
        
        """ Classification prediction - It classifies a customer based on a threshold: whether they are 
        postive or negative """
        
        pred_prob = self._glm_predict_pos_prob(input_sample)
        if self.threshold_glm is None:
            raise ValueError("Provide the threshold value in the class constructor")
        else:
            if pred_prob[sample] >= self.threshold_glm: # assign threshold manually
                return 1
            else:
                return 0

    def glm_sample_prob_pred(self, sample, input_sample):
        
        """ Probability prediction - It returns the probability that is greater than the threshold:
        The postive class or else it returns the negative class probability """
        
        pred_prob = self._glm_predict_pos_prob(input_sample)
        if self.threshold_glm is None:
            raise ValueError("Provide the threshold value in the class constructor")
        if pred_prob[sample] >= self.threshold_glm:
            return pred_prob[sample]
        else:
            return pred_prob[sample]

    def glm_binary_prediction(self, x_test):
         
        """ Prediction Function @ maximal threshold or user supplied threshold"""
        try:
            if self.threshold_glm is None:
                raise ValueError("Provide the threshold value in the class constructor")
        except ValueError as v:
            modelperf_logger.error(f"Exception of Type {v} occured",exc_info=True)
        predict_binary = self._glm_predict_pos_prob(x_test)        
        for i in range(x_test.shape[0]):
            if predict_binary[i] >= self.threshold_glm:
               predict_binary[i] = 1                   
            else: 
               predict_binary[i] = 0            
        return predict_binary

    def glm_overfitting_test(self, x_train, y_train, x_test, y_test, *thresholds):

        """ Overfitting performed specifically on Trainning and Testing sets
        This must be modified to be conducted on training and validation sets """

        train_precisions = []
        test_precisions = []
        train_recalls = []
        test_recalls = []

        for threshold in thresholds:
            y_prob_train = self._glm_predict_pos_prob(x_train)
            y_prob_test = self._glm_predict_pos_prob(x_test)

            y_pred_train = (y_prob_train > threshold).astype(int)
            print(y_pred_train)
            y_pred_test = (y_prob_test > threshold).astype(int)

            train_precision = precision_score(y_train, y_pred_train)
            test_precision = precision_score(y_test, y_pred_test)

            train_recall = recall_score(y_train, y_pred_train)
            test_recall = recall_score(y_test, y_pred_test)

            train_precisions.append(train_precision)
            test_precisions.append(test_precision)
            train_recalls.append(train_recall)
            test_recalls.append(test_recall)

        self.fig, (self.axs1, self.axs2)  = plt.subplots(1, 2)
        self.axs1.plot(thresholds, train_precisions, label = "Training Precision")
        self.axs1.plot(thresholds, test_precisions, label='Testing Precision')
        self.axs1.set_label('Thresholds')
        self.axs1.set_label('Precisions')
        self.axs1.legend()

        self.axs2.plot(thresholds, train_recalls, label = "Training Recalls")
        self.axs2.plot(thresholds, test_recalls, label='Testing Recalls')
        self.axs2.set_label('Thresholds')
        self.axs2.set_label('Reaclls')
        self.axs2.legend()
        return self.fig, train_precisions, train_recalls

    def glm_model_fit(self):

        """ Assessing model fit using chi-square from the regression output summary"""

        p_val = scipy.stats.chi2.pdf(loaded_model[0].pearson_chi2)
        return p_val

    def glm_roc_plot(self, x_test, y_test):
    
        """ Roc curve analytics and plot - Rocs points represent confusion matrices at varying 
        thresholds """
        
        self.fig, self.axs = plt.subplots(1,1)
        fpr, tpr, thresholds = roc_curve(y_test, self._glm_predict_pos_prob(x_test))
        self.axs.plot(fpr, tpr)
        super(ModelPerfomance, self)._plotting("Roc Curve", "fpr", "tpr")
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thres = thresholds[optimal_idx]
        return self.fig, optimal_thres
   
    def glm_perf_analytics(self, x_test, y_test):

        """ An Optimal threshold selected based on the number of accepted false postive rates to
        be used fo further assessment """

        try:
            if self.threshold_glm is None:
                raise ValueError("Provide the threshold value in the class constructor")
            plt.close('all')
            self.fig, self.axs = plt.subplots(1,1)
            y_bin= self.glm_binary_prediction(x_test)
            accuracy = accuracy_score(y_test, y_bin)
            f1 = f1_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_bin)
            threshold = self.threshold_dt
            data = pd.DataFrame({
            "Metric": ["threshold", "accuracy", "f1", "auc"],
            "Value": [threshold, accuracy, f1, auc]
            }) 
            sns.set_theme(style="ticks", color_codes=True)    
            sns.barplot(x="Metric", y="Value", data=data, ax=self.axs) 
            self.axs.set_title("GLM")
            return self.fig, threshold, accuracy, f1, auc            
        except ValueError as v:
            modelperf_logger.error(f"Exception of Type {v} occured",exc_info=True)

    def confusion_matrix_plot(self, x_test, y_test):
        
        """ confusion matrix plot at maximal threshold """
        
        def conf_plot():
            conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
            conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
            conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
            conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        self.fig, self.axs = plt.subplots(1,1)
        pred_bin = pd.Series(self.glm_binary_prediction(x_test))   
        conf_matrix = confusion_matrix(y_test, pred_bin)
        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_plot()
        return self.fig, conf_matrix

    def glm_probability_prediction(self, x_test):

        """ A wrapper to prodiuce list probability predictions """
         
        prediction_prob = [round(i,10) for i in self._glm_predict_pos_prob(x_test).tolist()]      
        return prediction_prob

    def glm_rev_perf_analytics(self, x_test):

        """ An Optimal threshold selected based on the number of accepted false postive rates to
        be used fo further assessment """

        fpr, tpr, thresholds = roc_curve(y_test, self.revised_logistic_model()[1].predict(x_test))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thres = thresholds[optimal_idx]
        y_prob= self.glm_binary_prediction(x_test)
        accuracy = accuracy_score(y_test, y_prob)
        f1 = f1_score(y_test, y_prob)
        return optimal_thres, accuracy, f1



