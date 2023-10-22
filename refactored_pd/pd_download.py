
import pandas as pd
import warnings

def data_download_sas(file_path):
    
    ''' Data Download function '''
    
    df_pd = pd.read_sas(file_path)

    return df_pd

def data_cleaning(file_path):
    
    ''' Various data cleaning functionalities '''
    
    dataframe = data_download_sas(file_path)
    data_types = dataframe.dtypes
    df_cat = dataframe.select_dtypes(object)
    df_float = dataframe.select_dtypes(float)
    for i in range(df_cat.shape[0]):
        for j in range(df_cat.shape[1]):
            if type(df_cat.iloc[i,j]) == bytes:                
                y = df_cat.iloc[i,j].decode("utf-8")
                df_cat.replace(df_cat.iloc[i,j], y, inplace=True)
            else:
                pass
    df_cat['PRODUCT'] = df_cat['PRODUCT'].replace('Others','OT')
    df_cat['NAT'] = df_cat['NAT'].replace('Others','RS')

    return data_types, df_cat, df_float