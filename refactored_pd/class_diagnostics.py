
""" 
    ==================
    Diagonostics Tests
    ==================
 
       Hypothesis Tests and Visual Plots:
     
         1. Quantile Residuals - Residuals for Discrete GLMs
         2. Breush Pagan Test - Heteroskedasticity of Variance
         3. Normal Residuals Test
         4. Durbin Watson Test - Test for Errors Serial Correlation
         5. Leverage Studentized Quantile Residuals
         6. Partial Residuals Plots
         7. Cooks Distance Quantile Residuals
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import logging

from scipy.stats import norm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from scipy.stats import probplot, normaltest
from math import sqrt
import statsmodels.api as sm
import pickle
import statsmodels.stats.diagnostic as sd

from class_modelperf import ModelPerfomance
from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

diagnostics_logger = logging.getLogger("class_diagnostics")
diagnostics_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
diagnostics_logger.addHandler(console_handler)
diagnostics_logger.info("MODEL DIAGNOSTICS ARE INCLUDED")
with open('glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class QuantileResiduals(ModelPerfomance, object):

    def __init__(self, custom_rcParams, x_test, y_test, threshold):
        
        super(QuantileResiduals, self).__init__(custom_rcParams, x_test, y_test, threshold)
        super(ModelPerfomance, self).__init__(custom_rcParams)

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in QuantileResiduals.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is class: {self.__class__.__name__}, and this is the public interface: {method_names}"

    def quantile_residuals(self):

        residuals = []
        try:
            if not isinstance(self.x_test, np.ndarray):
                raise TypeError("must be an instance of a numpy-ndarray")            
            predict_probability = super(QuantileResiduals, self).probability_prediction()
            if self.y_test.shape[0] is None:
                raise IndexError ("index empty")
            for i in range(self.y_test.shape[0]):
                if 0 <= self.threshold <= 1:
                    if (predict_probability[i] < self.threshold):
                        u_1 = np.random.uniform(low=0, high=predict_probability[i])
                        residuals.append(norm.ppf(u_1))
                    else:
                        u_2 = np.random.uniform(low=predict_probability[i], high=1)
                        residuals.append(norm.ppf(u_2))
                elif (self.threshold < 0 or self.threshold > 1):
                    raise ValueError("threshold outside bounds: [0-1]")
            quantile_residuals_series = pd.Series(residuals).round(2)

            return quantile_residuals_series

        except (TypeError, ValueError, IndexError) as e:
            diagnostics_logger.error(f"Exception of Type {e} occured",exc_info=True)

class ResidualsPlot(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in ResidualsPlot.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def plot_quantile_residuals(self):

        """ Residuals Plot """

        self.fig, self.axs = plt.subplots(1,1)
        try:
            quantile_residuals_series = super(ResidualsPlot,self).quantile_residuals()
            if quantile_residuals_series is None:
                raise ValueError ("residuals empty")
            self.axs.plot(quantile_residuals_series.index, quantile_residuals_series.values)
            super(ResidualsPlot,self)._plotting("humbu", "x", "y")

            return self.fig
        
        except ValueError as v:
            print("Error:", v)

            return None

class BreushPaganTest(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in BreushPaganTest.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def breush_pagan_quantile(self):

        """ Breush Pagan Test for Hetereskedasticity of variance """

        quantile_residuals_series = super(BreushPaganTest,self).quantile_residuals()
        try:
            if quantile_residuals_series is None:
                raise ValueError ("residuals empty")
            test = sd.het_breuschpagan(quantile_residuals_series, self.x_test)

            return test
        
        except ValueError as v:
            print("Error:", v)

            return None

class NormalityTest(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in NormalityTest.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def normality_test_quantile(self):

        """ normal test statistics """

        quantile_residuals_series = super(NormalityTest,self).quantile_residuals()
        normal_test = normaltest(quantile_residuals_series)

        return normal_test

    def plot_normality_quantile(self):

       """ normality plot"""

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super(NormalityTest,self).quantile_residuals()
       qqplot = stats.probplot(quantile_residuals_series, dist="norm")
       self.axs.plot(qqplot[0][0],qqplot[0][1], marker='o', linestyle='none')
       super(NormalityTest,self)._plotting("Normality Test", "x", "y")
        
       return self.fig

class DurbinWatsonTest(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in DurbinWatsonTest.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def durbin_watson_quantile(self):

        """ Durbin Watson Test for Residuals correlation range(1,5 - 2) """

        quantile_residuals_series = super(DurbinWatsonTest,self).quantile_residuals()
        durbin_watson_corr_test = durbin_watson(quantile_residuals_series)

        return durbin_watson_corr_test

class PartialPlots(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in PartialPlots.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def partial_plots_quantile(self, ind_var):

       """ Partial Plots - Residuals vs Features """

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super(PartialPlots,self).quantile_residuals()
       xlabel_name = ind_var.name
       self.axs.scatter(ind_var, quantile_residuals_series)
       super(PartialPlots,self)._plotting("Partial Plot", xlabel_name, "Residuals")
        
       return self.fig

class LevStudQuaRes(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in LevStudQuaRes.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def plot_lev_stud_quantile(self):

       """ Outliers and Influence """

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super(LevStudQuaRes,self).quantile_residuals()
       hat_matrix = np.round(loaded_model.get_hat_matrix_diag(),2)
       lev_stud_res = []
       for i in range(len(quantile_residuals_series)):            
           lev_stud_res.append(quantile_residuals_series[i]/(sqrt(1-hat_matrix[i])))
       self.axs.plot(pd.Series(lev_stud_res).index, pd.Series(lev_stud_res).values)
       super(LevStudQuaRes,self)._plotting("Leverage Studentised Residuals", "x", "y")
        
       return self.fig

class CooksDisQuantRes(QuantileResiduals, object):

    def __str__(self):

        pattern = re.compile(r'^_')
        method_names = []
        for name, func in CooksDisQuantRes.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def plot_cooks_dis_quantile(self):

        """ Cooks Distance Plot """

        self.fig, self.axs = plt.subplots(1,1)
        quantile_residuals_series = super(CooksDisQuantRes,self).quantile_residuals()
        hat_matrix = np.round(loaded_model.get_hat_matrix_diag(),2)
        d = []
        for i in range(len(quantile_residuals_series)):            
            d.append((quantile_residuals_series[i]**2/3000)*(hat_matrix[i]/(1-hat_matrix[i])))
        self.axs.plot(pd.Series(d).index, pd.Series(d).values)
        super(CooksDisQuantRes,self)._plotting("Cooks Distance", "x", "y")

        return self.fig