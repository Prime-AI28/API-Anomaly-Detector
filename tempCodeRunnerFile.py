import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def read_csv(file_path):
    data = pd.read_csv(file_path)
    data["DATE"] = pd.to_datetime(data["DATE"], format="%d-%m-%y")
    return data

def preprocess_data(data):
    df = data.copy()
    df["SUM_COUNT"] = df.groupby(["DATE", "APPNAME", "API"])["COUNT"].transform("sum")
    df_gp = df.drop_duplicates(subset=["DATE", "APPNAME", "API"])
    df_gp.drop(["COUNT", "STATUS"], axis=1, inplace=True)
    df_gp = df_gp.reset_index(drop=True)
    return df_gp

def calculate_zscores(df_gp):
    pro_df_dict_mean = {}
    pro_df_dict_std = {}
    pro_df_list = []

    applist = df_gp["APPNAME"].unique()
    apilist = []
    for i in applist:
        apilist.append(df_gp[df_gp["APPNAME"] == i]["API"].unique())

    for i, appname in enumerate(applist):
        for j, api_names in enumerate(apilist[i]):
            ind_df = df_gp[(df_gp["APPNAME"] == appname) & (df_gp["API"] == api_names)]
            zs = zscore(ind_df["SUM_COUNT"])
            mean = np.mean(zs)
            pro_df_dict_mean[appname + "-" + api_names] = mean
            sd = np.std(zs)
            pro_df_dict_std[appname + "-" + api_names] = sd

            ind_df["ZS"] = zs

            Outlier = pd.Series([0] * len(ind_df), index=ind_df.index)
            Outlier[zs >= 3] = 1
            Outlier[zs <= -3] = 1
            ind_df["OUTLIER"] = Outlier
            pro_df_list.append(ind_df)

    return pro_df_list

def encode_data(pro_df):
    pro_data = pro_df.copy()
    le = LabelEncoder()
    pro_data["APPNAME"] = le.fit_transform(pro_data["APPNAME"])
    pro_data["API"] = le.fit_transform(pro_data["API"])
    pro_data.sort_values(by=["DATE"], inplace=True)
    pro_data = pro_data.dropna()
    return pro_data

def train_test_split_data(pro_data):
    df_train = pro_data[pro_data["DATE"] < "2023-09-01"]
    df_test = pro_data[pro_data["DATE"] >= "2023-09-19"]
    df_train.drop(["DATE", "SUM_COUNT"], axis=1, inplace=True)
    df_test.drop(["DATE", "SUM_COUNT"], axis=1, inplace=True)

    an_train = df_train["OUTLIER"]
    an_test = df_test["OUTLIER"]

    df_train.drop(["OUTLIER"], axis=1, inplace=True)
    df_test.drop(["OUTLIER"], axis=1, inplace=True)

    return df_train, an_train, df_test, an_test

def build_isolation_forest(df_train, an_train, df_test):
    iso_forest = IsolationForest(contamination=0.06)
    iso_forest.fit(df_train, an_train)
    y_pred_iso = iso_forest.predict(df_test)
    y_pred_iso_map = np.where(y_pred_iso == -1, 1, 0)
    return y_pred_iso_map

def print_list(df_test, O_list):
    df_test["OUTLIER"] = O_list
    print(df_test)
    

def display_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    cm_iso = confusion_matrix(y_true, y_pred)
    print(cm_iso)

def main():
    data = read_csv("D:/Project/Base Models/API_Trend.csv")
    df_gp = preprocess_data(data)
    pro_df_list = calculate_zscores(df_gp)
    pro_data = encode_data(pd.concat(pro_df_list))
    df_train, an_train, df_test, an_test = train_test_split_data(pro_data)
    y_pred_iso_map = build_isolation_forest(df_train, an_train, df_test)
    display_results(an_test, y_pred_iso_map)
    print_list(df_test, y_pred_iso_map)

if __name__ == "__main__":
    main()
