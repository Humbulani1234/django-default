
import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm
from matplotlib import pyplot as plt

import pd_download
from class_traintest import OneHotEncoding
from class_base import Base
from class_missing_values import ImputationCat
import class_diagnostics
from class_modelperf import ModelPerfomance
from class_decision_tree import DecisionTree
from class_diagnostics import (ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
                               PartialPlots, LevStudQuaRes, CooksDisQuantRes, QuantileResiduals)

#----------------------------------------------------------------Data------------------------------------------------

with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

file_path = "refactored_pd/KGB.sas7bdat"
data_types, df_loan_categorical, df_loan_float = pd_download.data_cleaning(file_path)    
miss = ImputationCat(df_loan_categorical)
imputer_cat = miss.simple_imputer_mode()

custom_rcParams = {"figure.figsize": (9, 8), "axes.labelsize": 12}
threshold = -1
randomstate = 42

instance_mach = OneHotEncoding(custom_rcParams, imputer_cat, "machine")

x_test_orig = instance_mach.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]
conf_x_test = x_test_orig.reset_index(drop=True).iloc[0]
x_train_orig = instance_mach.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
y_test_orig = instance_mach.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]

instance_stats = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")

x_test = instance_stats.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]
ind_var = x_test_orig["CHILDREN"]
x_train = instance_stats.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]

x_test = sm.add_constant(x_test.values)
y_test = instance_stats.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]

m = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)

sample = 0
ccpalpha = 0
threshold_1=0.0019
threshold_2=0.0021

b = ResidualsPlot(custom_rcParams, x_test, y_test, threshold)
# print(b)
e = BreushPaganTest(custom_rcParams, x_test, y_test, threshold)
k = NormalityTest(custom_rcParams, x_test, y_test, threshold)
g = DurbinWatsonTest(custom_rcParams, x_test, y_test, threshold)
h = PartialPlots(custom_rcParams, x_test, y_test, threshold)
i = LevStudQuaRes(custom_rcParams, x_test, y_test, threshold)
j = CooksDisQuantRes(custom_rcParams, x_test, y_test, threshold)

c = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)
d = DecisionTree(custom_rcParams, imputer_cat, "machine", y_test_orig,
                 df_loan_float, df_loan_float["GB"], threshold, randomstate)
m = QuantileResiduals(custom_rcParams, x_test, y_test, threshold)
print(m)


#---------------------------------------------Execution-----------------------------------------

# m.roc_curve_analytics()
# plt.show()


