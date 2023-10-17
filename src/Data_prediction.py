import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from datetime import datetime, timedelta
import collections
collections.Callable = collections.abc.Callable

def prediction(df):
    df['dow'] = df['ds'].dt.dayofweek.astype(int)
    df['ds'] = df['ds'].values.astype(float)

    data = outlier_removal(df)

    X = data[['ds', 'dow']]
    Y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

    model = RandomForestRegressor(n_estimators=100, random_state=21)
    model.fit(X_train, y_train)

    data['ds'] = data['ds'].astype('datetime64[ns]')

    last_date = data['ds'].max()

    future_date = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='D')
    f_d_df = pd.DataFrame({'ds': future_date})
    f_d_df['dow'] = f_d_df['ds'].dt.dayofweek.astype(int)
    f_d_df['ds'] = f_d_df['ds'].values.astype(float)

    f_pred = model.predict(f_d_df)

    f_d_df['y'] = f_pred
    f_d_df['ds'] = f_d_df['ds'].astype('datetime64[ns]')
    f_d_df = f_d_df.drop(['dow'], axis=1)

    print(f_d_df)

    return f_d_df

def outlier_removal(df):
    iso_for = IsolationForest(contamination=0.01, random_state=21)
    y_pred = iso_for.fit_predict(df)
    df_no_outlier = df[y_pred == 1]
    return df_no_outlier