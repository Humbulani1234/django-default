
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

# ----------------------------------------------------Base Class-----------------------------------------------------------

with open('refactored_pd/glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class QuantileResiduals(ModelPerfomance):

    def quantile_residuals(self):

        residuals = []

        try:

            if not isinstance(self.x_test, np.ndarray):

                raise TypeError("must be an instance of a numpy-ndarray")
            
            self.predict_probability = super().probability_prediction()

            if self.y_test.shape[0] is None:

                raise IndexError ("index empty")

            for i in range(self.y_test.shape[0]):

                if 0 <= self.threshold <= 1:

                    if (self.predict_probability[i] < self.threshold):

                        u_1 = np.random.uniform(low=0, high=self.predict_probability[i])
                        residuals.append(norm.ppf(u_1))

                    else:

                        u_2 = np.random.uniform(low=self.predict_probability[i], high=1)
                        residuals.append(norm.ppf(u_2))

                elif (self.threshold < 0 or self.threshold > 1):

                    raise ValueError("threshold outside bounds: [0-1]")

            quantile_residuals_series = pd.Series(residuals).round(2)

            return quantile_residuals_series

        except (TypeError, ValueError, IndexError) as e:

            print("Error:", e)

            return None

#------------------------------------------------------------Residuals Plot---------------------------------------

class ResidualsPlot(QuantileResiduals):

    def plot_quantile_residuals(self):

        """ Residuals Plot """

        self.fig, self.axs = plt.subplots(1,1)

        try:

            quantile_residuals_series = super().quantile_residuals()

            if quantile_residuals_series is None:

                raise ValueError ("residuals empty")

            self.axs.plot(quantile_residuals_series.index, quantile_residuals_series.values)
            super().plotting("humbu", "x", "y")

            return self.fig
        
        except ValueError as v:

            print("Error:", v)

            return None

# -------------------------------------------------Breush Pagan Test---------------------------------------------------

class BreushPaganTest(QuantileResiduals):


    def breush_pagan_quantile(self):

        """ Breush Pagan Test for Hetereskedasticity of variance """

        quantile_residuals_series = super().quantile_residuals()

        try:

            if quantile_residuals_series is None:
                raise ValueError ("residuals empty")

            self.test = sd.het_breuschpagan(quantile_residuals_series, self.x_test)

            return self.test
        
        except ValueError as v:

            print("Error:", v)

            return None

# ------------------------------------------------------Normality Test-----------------------------------------------

class NormalityTest(QuantileResiduals):

    def normality_test_quantile(self):

        """ normal test statistics """

        quantile_residuals_series = super().quantile_residuals()
        self.normal_test = normaltest(quantile_residuals_series)

        return self.normal_test

    def plot_normality_quantile(self):

       """ normality plot"""

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super().quantile_residuals()
       self.qqplot = stats.probplot(quantile_residuals_series, dist="norm")
       self.axs.plot(self.qqplot[0][0],self.qqplot[0][1], marker='o', linestyle='none')
       super().plotting("Normality Test", "x", "y")
        
       return self.fig

# ------------------------------------------------Durbin Watson Test-----------------------------------------------------

class DurbinWatsonTest(QuantileResiduals):

    def durbin_watson_quantile(self):

        """ Durbin Watson Test for Residuals correlation range(1,5 - 2) """

        quantile_residuals_series = super().quantile_residuals()
        self.durbin_watson_corr_test = durbin_watson(quantile_residuals_series)

        return self.durbin_watson_corr_test

# ----------------------------------------------Partial Plots-------------------------------------------------------

class PartialPlots(QuantileResiduals):

    def partial_plots_quantile(self, ind_var):

       """ Partial Plots - Residuals vs Features """

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super().quantile_residuals()
       self.xlabel_name = ind_var.name
       self.axs.scatter(ind_var, quantile_residuals_series)
       super().plotting("Partial Plot", self.xlabel_name, "Residuals")
        
       return self.fig

# -------------------------------------------------Leverage Studentised Residuals-----------------------------------------

class LevStudQuaRes(QuantileResiduals):

    def plot_lev_stud_quantile(self):

       """ Outliers and Influence """

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super().quantile_residuals()
       hat_matrix = np.round(loaded_model.get_hat_matrix_diag(),2)
       self.lev_stud_res = []

       for i in range(len(quantile_residuals_series)):
            
        self.lev_stud_res.append(quantile_residuals_series[i]/(sqrt(1-hat_matrix[i])))

       self.axs.plot(pd.Series(self.lev_stud_res).index, pd.Series(self.lev_stud_res).values)
       super().plotting("Leverage Studentised Residuals", "x", "y")
        
       return self.fig

# -------------------------------------------------Cooks Distance Residuals---------------------------------------------

class CooksDisQuantRes(QuantileResiduals):

    def plot_cooks_dis_quantile(self):

        """ Cooks Distance Plot """

        self.fig, self.axs = plt.subplots(1,1)
        quantile_residuals_series = super().quantile_residuals()
        hat_matrix = np.round(loaded_model.get_hat_matrix_diag(),2)
        self.d = []

        for i in range(len(quantile_residuals_series)):
            
            self.d.append((quantile_residuals_series[i]**2/3000)*(hat_matrix[i]/(1-hat_matrix[i])))

        self.axs.plot(pd.Series(self.d).index, pd.Series(self.d).values)
        super().plotting("Cooks Distance", "x", "y")

        return self.fig