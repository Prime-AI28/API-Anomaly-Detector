import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import ML_model as ad


def Anomaly_detection(df):

    pro_df_list = []
    apilist = []
    pro_df_dict = {}
    predict_anomaly_list = []
    final_list = []

    df['SUM_COUNT'] = df.groupby(['DATE','APPNAME', 'API'])['COUNT'].transform('sum')
    df_gp = df.drop_duplicates(subset=['DATE','APPNAME', 'API'])
    df_gp.drop(['COUNT', 'STATUS'], axis=1, inplace=True)
    df_gp = df_gp.reset_index(drop=True)

    df_gp.sort_values(by=['APPNAME'], inplace=True)
    applist = df_gp['APPNAME'].unique()
    for i in applist:
        apilist.append(df_gp[df_gp['APPNAME']==i]['API'].unique())

    for i, appname in enumerate(applist):
        for j, api_names in enumerate(apilist[i]):
            
            ind_df = df_gp[(df_gp['APPNAME']==appname) & (df_gp['API']==api_names)]
            zs = zscore(ind_df['SUM_COUNT'])
            ind_df['ZS'] = zs

            Outlier = pd.Series([0]*len(ind_df), index=ind_df.index)
            Outlier[zs>=3] = 1
            Outlier[zs<=-3] = 1
            ind_df['OUTLIER'] = Outlier
            pro_df_list.append(ind_df)

    pro_df = pd.concat(pro_df_list)

    for b in range(0,len(pro_df_list)):
        app_name = pro_df_list[b]['APPNAME'].unique()
        api_name = pro_df_list[b]['API'].unique()
        en = app_name + '-' + api_name
        pro_df_dict[b] = en

    # list out keys and values separately
    key_list = list(pro_df_dict.keys())
    val_list = list(pro_df_dict.values())

    for a in range(0, len(pro_df_list)):
        pro_data = pro_df_list[a].copy()
        pro_data,dl =  feature_engeenering(pro_data)

        df_train = pro_data[pro_data['DATE']<dl[-1]]
        df_test = pro_data[(pro_data['DATE']==dl[-1])]

        df_train.drop(['DATE'], axis=1, inplace=True)
        df_train.drop(['ZS'], axis=1, inplace=True)
        df_test.drop(['DATE'], axis=1, inplace=True)
        df_test.drop(['ZS'], axis=1, inplace=True)

        an_train = df_train['OUTLIER']
        an_test = df_test['OUTLIER']
        
        df_train.drop(['OUTLIER'], axis=1, inplace=True)
        df_test.drop(['OUTLIER'], axis=1, inplace=True)

        if(len(df_train)!=0) & (len(df_test)!=0):
            iso_forest = IsolationForest(contamination=0.01)
            iso_forest.fit(df_train, an_train)

            y_pred = iso_forest.predict(df_test)
            y_pred_iso_map = np.where(y_pred == -1, 1, 0)

            df_final = df_test.copy()
            df_final['OUTLIER'] = y_pred_iso_map
            df_f_w = df_final[df_final['OUTLIER']==1]

            ind_df_f = pro_df_list[a].copy()
            ind_df_f = ind_df_f[ind_df_f['DATE']==dl[-1]]

            ind_df_f.drop(['OUTLIER'], axis=1, inplace=True)
            ind_df_f['OUTLIER'] = df_final.loc[ind_df_f.index]['OUTLIER']
            li = ind_df_f['APPNAME'].index

            if not (ind_df_f[ind_df_f['DATE'] == '2023-09-19'][ind_df_f['OUTLIER'] == 1].empty):
                final_list.append(pro_df_dict[a])

    return final_list, 'Model trained on Isolation forest'
         
      

def feature_engeenering(pro_data):
    # Implement the logic to train the anomaly detection model
    le = LabelEncoder()
    pro_data['APPNAME'] = le.fit_transform(pro_data['APPNAME'])
    pro_data['API'] = le.fit_transform(pro_data['API'])
    pro_data.sort_values(by=['DATE'], inplace=True)
    dl = pro_data['DATE'].unique()
    pro_data = pro_data.dropna()
    return pro_data, dl



