"""
  ==============================
  MCAR adhoc tests vs MNAR, MAR
  ==============================

  ======
  Plots
  ======
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import logging

from pd_download import data_cleaning_pd, data_cleaning_ead


class ImputationCat:
    def __init__(self, df_cat):
        self.df_cat = df_cat

    def __str__(self):
        pattern = re.compile(r"^_")
        method_names = []
        for name, func in ImputationCat.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def simple_imputer_mode(self):
        """Simple Imputation"""

        df_cat_mode = self.df_cat.copy(True)
        mode_imputer = SimpleImputer(strategy="most_frequent")
        df_cat_mode.iloc[:, :] = mode_imputer.fit_transform(df_cat_mode)

        return df_cat_mode

    def KNN_Imputation(self):
        """KNN imputation"""

        dataframe_array = df_cat.to_numpy().astype(float)
        dataframe_impute_KNN = impy.fast_knn(dataframe_array)

        return pd.DataFrame(dataframe_impute_KNN)

    def _ordinal_encode_nan(
        self, independent_series, dataframe
    ):  # for one column, then procedural
        """Ordinal Encoding with missing values"""

        y = OrdinalEncoder()  # instatiate ordinal encoder class
        name = independent_series  # pass in the independent series for a missing column, (name = name of column)
        name_not_null = independent_series[
            independent_series.notnull()
        ]  # removes null values from column
        reshaped_vals = name_not_null.values.reshape(
            -1, 1
        )  # extract series values only and reshape them for
        encoded_vals = y.fit_transform(reshaped_vals)  # function takes in array
        dataframe.loc[
            independent_series.notnull(), independent_series.name
        ] = np.squeeze(encoded_vals)

        return dataframe

    def concatenate_total_df(self, dataframefloat, dataframecategorical):
        """oncatenate the imputed dataframes(categorical/float)
        into one total dataframe for further analysis"""

        df_total_no_missing = pd.concat([dataframefloat, dataframecategorical], axis=1)

        return df_total_no_missing


if __name__ == "__main__":
    file_path = "KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning_pd(file_path)
    miss = ImputationCat(df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()
    print(imputer_cat)
    print(df_loan_float)

    # Alternative one hot encoding scheme

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_data = encoder.fit_transform(imputer_cat[["TITLE"]])
    column_names = encoder.get_feature_names_out(["TITLE"])
    encoded_data = pd.DataFrame(encoded_data, columns=column_names)
    print(encoded_data)
    reference_category = encoder.categories_[0][0]
    print(reference_category)
