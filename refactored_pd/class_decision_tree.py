

"""

    ===================================
    MODEL ALTERNATIVES - Decision Tree
    ===================================

    We investigate Decision Trees as a model alternative to GLM - Binomial

    ==========
    Base Tree
    ==========

    ===============
    Fit a base tree
    ===============


"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

# -----------------------------------------------------------------Class DecisionTree------------------------------------------------

class DecisionTree(OneHotEncoding):

    """ Fit a base tree """

    def __init__(self, custom_rcParams, df_nomiss_cat, type_, y_test,
                  df_loan_float, target, threshold, randomstate):

        super().__init__(custom_rcParams, df_nomiss_cat, type_)
        
        # self.fig, self.axs = plt.subplots(1,1)  
        self.df_loan_float = df_loan_float
        self.target = target
        self.randomstate = randomstate
        self.x_train = super().split_xtrain_ytrain(self.df_loan_float, self.target)[0]
        self.y_train = super().split_xtrain_ytrain(self.df_loan_float, self.target)[2]

    def dt_classification_fit(self, ccpalpha):
        
        ''' DT Classification fit '''
        
        clf_dt = DecisionTreeClassifier(random_state=self.randomstate, ccp_alpha=ccpalpha)
        clf_dt = clf_dt.fit(self.x_train, self.y_train)
        
        return clf_dt

    def dt_probability_prediction(self, sample, x_test1, ccpalpha):
        
        ''' Base tree prediction '''
        
        predict_dt = self.dt_classification_fit(ccpalpha).predict_proba(x_test1).round(10)

        if predict_dt[sample][1] >= 0.47:

            return predict_dt[sample][1]

        else:

            return predict_dt[sample][1]

    def dt_confusion_matrix_plot(self, x_test, y_test, ccpalpha):

        """ Base tree Confusion matrix """

        self.fig, self.axs = plt.subplots(1,1)  

        predict_dt_series = pd.Series(self.dt_classification_fit(ccpalpha).predict(x_test))       
        conf_matrix = confusion_matrix(y_test, predict_dt_series)
        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
        conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
        conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
        conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        return self.fig

    def plot_dt(self, ccpalpha):

        """ Base tree plot """

        self.fig, self.axs = plt.subplots(1,1)  

        clf_dt = self.dt_classification_fit(ccpalpha)
        plot_tree(clf_dt, filled = True, rounded = True, 
                   feature_names = self.x_train.columns.tolist(), ax = self.axs)   

        return self.fig

    def pruning(self, ccpalpha):

        """ Extracting alphas for pruning """
        
        clf_dt = self.dt_classification_fit(ccpalpha)
        path = clf_dt.cost_complexity_pruning_path(self.x_train, self.y_train) 
        ccp_alphas = path.ccp_alphas 
        ccp_alphas = ccp_alphas[:-1] 
        
        return ccp_alphas

    def cross_validate_alphas(self, ccpalpha):
        
        """ Cross validation for best alpha """

        self.fig, self.axs = plt.subplots(1,1)  

        alpha_loop_values = []        
        ccp_alphas = self.pruning(ccpalpha)

        for ccp_alpha in ccp_alphas:

            clf_dt = DecisionTreeClassifier(random_state=self.randomstate, ccp_alpha=ccp_alpha)
            scores = cross_val_score(clf_dt, self.x_train, self.y_train, cv=5)
            alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
        
        alpha_results = pd.DataFrame(alpha_loop_values, columns=["alpha", "mean_accuracy", "std"])
        alpha_results.plot(ax = self.axs, x = "alpha", y = "mean_accuracy", yerr = "std", marker = "o" , linestyle = "--")
        self.axs.spines["top"].set_visible(False)  
        self.axs.spines["right"].set_visible(False) 
        
        return alpha_results, self.fig

    def ideal_alpha(self, ccpalpha, threshold_1, threshold_2):
        
        """ Extraction of ideal alpha """

        alpha_results = self.cross_validate_alphas(ccpalpha)[0]       
        ideal_ccp_alpha = alpha_results[(alpha_results["alpha"] > threshold_1) & (alpha_results["alpha"] < threshold_2)]["alpha"]
        ideal_ccp_alpha = ideal_ccp_alpha.values.tolist()
        
        return ideal_ccp_alpha[0]

    def dt_pruned_alpha(self, ccpalpha, threshold_1, threshold_2):

        """ Ideal alpha value for pruning the tree """

        ideal_ccp_alpha = self.ideal_alpha(ccpalpha, threshold_1, threshold_2)

        return ideal_ccp_alpha

    def dt_pruned_fit(self, ccpalpha, threshold_1, threshold_2):

        """ Pruned tree fitting """

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_clf_dt = self.ideal_alpha(ideal_ccp_alpha)

        return pruned_clf_dt

    def dt_pruned_prediction(self, ccpalpha, threshold_1, threshold_2, sample, x_test1):

        """ Prediction and perfomance analytics """

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_predict_dt = self.dt_probability_prediction(sample, x_test1, ideal_ccp_alpha)

        return pruned_predict_dt

    def dt_pruned_confmatrix(self, ccpalpha, threshold_1, threshold_2, x_test, y_test):

        """ Confusion matrix plot """

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_confusion_matrix = self.dt_confusion_matrix_plot(x_test, y_test, ideal_ccp_alpha)

        return pruned_confusion_matrix

    def dt_pruned_tree(self, ccpalpha, threshold_1, threshold_2):

        """ Plot final tree """

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_plot_tree = self.plot_dt(ideal_ccp_alpha)

        return pruned_plot_tree  
