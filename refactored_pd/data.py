
import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

import pd_download
from class_traintest import OneHotEncoding
from class_base import Base
from class_missing_values import ImputationCat
import class_diagnostics
from class_modelperf import ModelPerfomance
from class_decision_tree import DecisionTree
from class_diagnostics import (ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
                               PartialPlots, LevStudQuaRes, CooksDisQuantRes, QuantileResiduals)
from class_lgclassifier import LogRegression
from class_comparison import ModelComparison
from class_clustering_pd import ClusterProbability
from class_ols_ridge import RidgeAndOLS

pd.set_option('display.max_columns', 1200)

#----------------------------------------------------------------Data------------------------------------------------

with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

file_path = "refactored_pd/KGB.sas7bdat"
data_types, df_loan_categorical, df_loan_float = pd_download.data_cleaning_pd(file_path)    
miss = ImputationCat(df_loan_categorical)
imputer_cat = miss.simple_imputer_mode()

custom_rcParams = {"figure.figsize": (9, 8), "axes.labelsize": 12}
threshold = 0.7
randomstate = 42

sample = 0
ccpalpha = 0
threshold_1=0.0019
threshold_2=0.0021
thresholds = np.arange(0.1, 0.9, 0.05)

# #-----------------------------------------------Statistics--------------------------------------------

instance_stats = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")

m = ModelPerfomance(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"], randomstate, threshold)
q = ClusterProbability(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"], randomstate, threshold)

x_train_glm_o = sm.add_constant(m.x_train_glm.values, has_constant='add')
x_test_glm_o = sm.add_constant(m.x_test_glm.values, has_constant='add')
# instance_stats = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")

# x_test_o = instance_stats.train_val_test(df_loan_float, target=df_loan_float["GB"])[4] # for revised model test
# # print(x_test_o.shape)
# y_test_o = instance_stats.train_val_test(df_loan_float, target=df_loan_float["GB"])[5]
# # ind_var = x_test_orig["CHILDREN"]
# x_train_o = instance_stats.train_val_test(df_loan_float, target=df_loan_float["GB"])[0] # for revised model fit, the _o
# # print(x_train_o.shape)
# y_train_o = instance_stats.train_val_test(df_loan_float, target=df_loan_float["GB"])[1]
# # print(y_train_o)

# Data passed into the class object for initialisation

# x_test_o = sm.add_constant(x_test_o.values, has_constant='add')
# x_train_o = sm.add_constant(x_train_o.values, has_constant='add')
# # print(x_test_o)
# # x_train = sm.add_constant(x_train_o.values)
# # y_test = instance_stats.train_val_test(df_loan_float, target=df_loan_float["GB"])[5]

# print(m.glm_reg_equ())
# print(m.glm_binary_prediction(x_test_o))
# print(m.glm_perf_analytics(x_test_o, y_test_o))
# print(m.confusion_matrix_plot(x_test_o, y_test_o))
# plt.show()
# print(m.glm_overfitting_test(x_train_o, y_train_o, x_test_o, y_test_o, *np.arange(0.1, 0.9, 0.05)))
# plt.show()

#Data for revised logistic regression model

# x_test_re = x_test_o[['AGE', 'TEL' ,'NMBLOAN', 'LOANS', 'Car', 'Mastercard/Euroc']]
# x_test_re = sm.add_constant(x_test_re.values)
# print(m.optimal_threshold(x_test_re)[1])

# diagnostics

ind_var = m.x_test_glm["CHILDREN"]

b = ResidualsPlot(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
b.x_test_glm = sm.add_constant(b.x_test_glm.values, has_constant='add')
# print(b.quantile_residuals(b.x_test_glm))
e = BreushPaganTest(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
# print(e.breush_pagan_quantile(b.x_test_glm))
k = NormalityTest(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
g = DurbinWatsonTest(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
h = PartialPlots(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
i = LevStudQuaRes(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)
j = CooksDisQuantRes(custom_rcParams, imputer_cat, "statistics",
                 df_loan_float, df_loan_float["GB"],randomstate, threshold)

#---------------------------------------------Decision Trees-----------------------------------------

d = DecisionTree(custom_rcParams, imputer_cat, "machine",
                 df_loan_float, df_loan_float["GB"], GridSearchCV, randomstate, onehot=True, threshold=0.47)

# print(d.dt_pruned_perf_analytics(ccpalpha, threshold_1, threshold_2, d.x_test_dt, d.y_test_dt))
# print(d.dt_binary_prediction(x_test_orig, ccpalpha))
# print(d.dt_sample_pruned_prob(ccpalpha, threshold_1, threshold_2, 15, x_test_orig))
# print(d.lg_pruned_overfitting(ccpalpha, threshold_1, threshold_2, x_train_orig, y_train_orig, x_test_orig, y_test_orig,
                             # *np.arange(0.1, 0.9, 0.05)))
# print(d.dt_pruned_conf_matrix(ccpalpha, threshold_1, threshold_2, x_test_orig, y_test_orig, threshold))
# print(d.dt_pruned_feature_imp(ccpalpha, threshold_1, threshold_2))

#------------------------------------------------Logistic Regression-----------------------------------

s = LogRegression(custom_rcParams, imputer_cat, "machine",
                 df_loan_float, df_loan_float["GB"], GridSearchCV, randomstate, onehot=True, threshold=0.47)

# print(s.lg_overfitting_test(x_train_orig, y_train_orig, x_test_orig, y_test_orig, *np.arange(0.1, 0.9, 0.05)))
# print(s.lg_binary_prediction(x_test_orig))
# print(s.lg_perf_analytics(x_test_orig, y_test_orig))
# print(s.lg_sample_prob_pred(15, x_test_orig))
# plt.show()
# print(d.dt_pruned_feature_imp(ccpalpha, threshold_1, threshold_2))
# print(s.sgd_view_coef())
# print(d.dt_confusion_matrix_plot(x_test_orig, y_test_orig, threshold, ccpalpha))
# print(d.dt_classification_fit(ccpalpha).feature_importances_)

#-----------------------------------Comparison----------------------------------------------

o = ModelComparison(custom_rcParams, imputer_cat, "statistics", "machine",
                    df_loan_float, df_loan_float["GB"], GridSearchCV, randomstate, onehot=True, threshold=0.47)
# print(o.cmp_performance_metrics(ccpalpha, threshold_1, threshold_2, threshold))
# print(o.cmp_overfitting(ccpalpha, threshold_1, threshold_2, *np.arange(0.1, 0.9, 0.05)))
# o.cmp_confusion_matrix_plot(ccpalpha, threshold_1, threshold_2, threshold)
# plt.show()

#--------------------------------------------EAD LINEAR REGRESSION-------------------------------------------------

# file_path = "../data/cohort1.sas7bdat"
# data_types, df_loan_categorical, df_loan_float = pd_download.data_cleaning(file_path)    
# miss = ImputationCat(df_loan_categorical)
# imputer_cat = miss.simple_imputer_mode()
# print(df_loan_float)

#------------------------------------------------EAD LOGISTIC REGRESSION---------------------------------------------

# file_path = "../data/cohort1.sas7bdat"
# file_path = "../data/lgd_data_default.sas7bdat"
# data_types, df_loan_categorical, df_loan_float = pd_download.data_cleaning_ead(file_path)    
# miss = ImputationCat(df_loan_categorical)
# imputer_cat = miss.simple_imputer_mode()
# # print(df_loan_float)

# r = RidgeAndOLS(custom_rcParams, imputer_cat, "statistics", "machine",
#                 df_loan_float, df_loan_float["GB"], randomstate, Ridge, sm.OLS)
# print(r.OLS())


