import pandas as pd

def preprocessing(df):
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
    dates = sorted(df['DATE'].unique())
    data = feature_engineering(df)
    return df, dates

def feature_engineering(df):
    df['SUM_COUNT'] = df.groupby(['DATE','APPNAME', 'API'])['COUNT'].transform('sum')
    df.drop_duplicates(subset=['DATE','APPNAME', 'API'], inplace=True)
    df_n = feature_selection(df)
    return df_n

def feature_selection(df):
    df.drop(['COUNT', 'STATUS'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['DATE'], inplace=True)
    return df
