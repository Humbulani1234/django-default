
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from typing import Type
from sklearn.metrics import mean_squared_error

from class_traintest import OneHotEncoding

class RidgeAndOLS(OneHotEncoding, object):

    def __init__(self, custom_rcParams, df_nomiss_cat, type_1, type_2,
                 df_loan_float, target, randomstate, ridge:Type[Ridge], ols:Type[sm.OLS]):

        self.ols = ols
        self.ridge = ridge

        """ OLS datasets """

        OneHotEncoding.__init__(self, custom_rcParams, df_nomiss_cat, type_1, randomstate, False)

        self.df_loan_float_ols = df_loan_float
        self.target_ols = target
        self.random_state_ols = self.random_state_one

        self.x_train_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[0]
        self.y_train_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[1]
        self.x_val_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[2]
        self.y_val_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[3]
        self.x_test_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[4]
        self.y_test_ols = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_ols, self.target_ols)[5]

        self.x_train_ols = sm.add_constant(self.x_train_ols, has_constant="add")
        self.x_test_ols = sm.add_constant(self.x_test_ols, has_constant="add")

        """ RIDGE datasets """

        OneHotEncoding.__init__(self, custom_rcParams, df_nomiss_cat, type_2, randomstate)

        self.df_loan_float_rg = df_loan_float
        self.target_rg = target
        self.random_state_rg = self.random_state_one

        self.x_train_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[0]
        self.y_train_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[1]
        self.x_val_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[2]
        self.y_val_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[3]
        self.x_test_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[4]
        self.y_test_rg = super(RidgeAndOLS, self).train_val_test(self.df_loan_float_rg, self.target_rg)[5]

    def fit_ols(self):

        model = self.ols(self.y_train_ols, self.x_train_ols).fit()
        return model.summary(), model

    def OLS(self):
        model = LinearRegression().fit(self.x_train_rg, self.y_train_rg)
        return model

    def fit_ridge(self):

        alphas = np.logspace(-6, 6, 13)  # Example values spanning several orders of magnitude
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(self.ridge(), param_grid, cv=5)  # 5-fold cross-validation
        grid_search.fit(self.x_train_rg, self.y_train_rg)
        best_alpha = grid_search.best_params_['alpha']
        best_ridge = self.ridge(alpha=best_alpha)
        ridge_final = best_ridge.fit(self.x_train_rg, self.y_train_rg)
        return ridge_final
        
    def alpha_regime_ridge(self, *alphas):

        """ Method to investigate which values of alpha to focus on - run at the beginning and after 
        the regime determination, fit ridge with those alphas and still select the best model from the regime """

        train_mse = []
        val_mse = []
        for alpha in alphas:
            ridge = self.ridge(alpha=alpha)
            ridge_final = ridge.fit(self.x_train_rg, self.y_train_rg)
            ridge_train_pred = ridge.predict(self.x_train_rg)
            ridge_mse_train = mean_squared_error(self.y_train_rg, ridge_train_pred)
            
            ridge_val_pred = self.fit_ridge.predict(self.x_val_rg)
            ridge_mse_val = mean_squared_error(self.y_val_rg, ridge_train_pred)

            train_mse.append(ridge_mse_train)
            val_mse.append(ridge_mse_val)
        
        plt.close('all')
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

    def comparison(self):

        ols_predictions = ols_model.predict(X_test)
        ridge_predictions = ridge_model.predict(X_test)

        # Calculate the MSE
        ols_mse = mean_squared_error(y_test, ols_predictions)
        ridge_mse = mean_squared_error(y_test, ridge_predictions)



