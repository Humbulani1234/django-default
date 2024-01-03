import pandas as pd
import warnings


def data_download_sas(file_path):
    """Data Download function"""

    df_pd = pd.read_sas(file_path)
    return df_pd


def _cleaning(file_path):
    """Common data cleaning functionality"""

    dataframe = data_download_sas(file_path)
    data_types = dataframe.dtypes
    df_cat = dataframe.select_dtypes(object)
    df_float = dataframe.select_dtypes(float)
    for i in range(df_cat.shape[0]):
        for j in range(df_cat.shape[1]):
            if type(df_cat.iloc[i, j]) == bytes:
                y = df_cat.iloc[i, j].decode("utf-8")
                df_cat.replace(df_cat.iloc[i, j], y, inplace=True)
            else:
                pass
    return data_types, df_cat, df_float


def data_cleaning_pd(file_path):
    """Various data cleaning functionalities"""

    data_types, df_cat, df_float = _cleaning(file_path)
    df_cat["PRODUCT"] = df_cat["PRODUCT"].replace("Others", "OT")  # Naming cleaning
    df_cat["NAT"] = df_cat["NAT"].replace("Others", "RS")  # Naming cleaning
    df_float = df_float.drop(
        labels=["_freq_"], axis=1
    )  # same as GB, only 30.0 replaces 0.0 in GB
    return data_types, df_cat, df_float


def data_cleaning_ead(file_path):
    """Various data cleaning functionalities"""

    data_types, df_cat, df_float = _cleaning(file_path)
    df_float = df_float.drop(columns=["Cohort", "ID"])
    return data_types, df_cat, df_float
