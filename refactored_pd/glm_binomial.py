
import statsmodels.api as sm
import pandas as pd
import pickle
import scipy

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

def glm_binomial_fit(y_train, x_train):
    
    ''' GLM - Logistic regression model fit '''

    try:
        if y_train is list:
            raise TypeError("target not an instance of a list") 
    except TypeError as t:
        print("Error:", t)
    else:        
        x_train = sm.add_constant(x_train.values)
        glm_binom = sm.GLM(y_train, x_train, family=sm.families.Binomial())   
        res = glm_binom.fit()
        return res.summary(), res

if __name__ == "__main__":

    file_path = "./KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
    miss = ImputationCat(df_cat=df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()

    custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    instance = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")
    
    x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
    y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
    y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
    x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

    x_test = sm.add_constant(x_test.values)

    y_train_shape = y_train.values.reshape(-1,1)

    model = (glm_binomial_fit(y_train_shape, x_train))

    with open('glm_binomial.pkl','wb') as file:
        pickle.dump(model, file)
        