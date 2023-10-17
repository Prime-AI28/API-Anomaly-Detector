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

def clean_data(df):
    # Check for missing values
    df.isnull().sum()
    # Handle missing values (e.g., drop rows with missing values)
    df.dropna(inplace=True)
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
    df.rename(columns={'APPNAME': 'Application', 'API': 'API_Version', 'STATUS': 'Status', 'COUNT': 'Request_Count'}, inplace=True) 

    return df


def time_series_data(updf):
    df = clean_data(updf)
    total_unpro_df = df[df['Status'] != 'SE']
    data__total = total_unpro_df.groupby(['DATE']).sum().reset_index()
    data__total.drop(columns=['API_Version', 'Application', 'Status'], inplace=True)
    data__total.rename(columns={ 'DATE':'ds','Request_Count': 'y'}, inplace=True) 
    
    return data__total
