import pandas as pd
import numpy as np
from scipy.stats import zscore

def Zscore_calc(df):
    data_list = []
    for i in df['APPNAME'].unique():
        for j in df[df['APPNAME']==i]['API'].unique():
            ind_df = df[(df['APPNAME']==i) & (df['API']==j)]
            zs = zscore(ind_df['SUM_COUNT'])
            ind_df['ZS'] = zs
            Outlier = pd.Series([0]*len(ind_df), index=ind_df.index)
            Outlier[zs>=3] = 1
            Outlier[zs<=-3] = 1
            ind_df['OUTLIER'] = Outlier
            data_list.append(ind_df)
    final_df = pd.concat(data_list)
    return final_df

def Anomalies(df, dl):
    data = Zscore_calc(df)
    an_list = data[(data['DATE'] == dl)][data['OUTLIER'] == 1][['APPNAME', 'API']]
    return an_list