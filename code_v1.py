import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv("D:\Project\Base Models\API_Trend.csv")
data["DATE"] = pd.to_datetime(data["DATE"], format="%d-%m-%y")

df = data.copy()

df["SUM_COUNT"] = df.groupby(["DATE", "APPNAME", "API"])["COUNT"].transform("sum")
df_gp = df.drop_duplicates(subset=["DATE", "APPNAME", "API"])
df_gp.drop(["COUNT", "STATUS"], axis=1, inplace=True)
df_gp = df_gp.reset_index(drop=True)

df_gp.sort_values(by=["APPNAME"], inplace=True)
applist = df_gp["APPNAME"].unique()
apilist = []
for i in applist:
    apilist.append(df_gp[df_gp["APPNAME"] == i]["API"].unique())

pro_df_dict_mean = {}
pro_df_dict_std = {}


pro_df_list = []
count = 0
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

pro_df = pd.concat(pro_df_list)


pro_df_dict = {}
for b in range(0, len(pro_df_list)):
    app_name = pro_df_list[b]["APPNAME"].unique()
    api_name = pro_df_list[b]["API"].unique()
    en = app_name + "-" + api_name
    pro_df_dict[b] = en
print(pro_df_dict)

# list out keys and values separately
key_list = list(pro_df_dict.keys())
val_list = list(pro_df_dict.values())

pro_data = pro_df.copy()
le = LabelEncoder()
pro_data["APPNAME"] = le.fit_transform(pro_data["APPNAME"])
pro_data["API"] = le.fit_transform(pro_data["API"])
pro_data.sort_values(by=["DATE"], inplace=True)
dl = pro_data["DATE"].unique()

print(pro_data.isnull().sum())
pro_data = pro_data.dropna()

df_train = pro_data[pro_data["DATE"] < "2023-09-01"]
df_test = pro_data[pro_data["DATE"] >= "2023-09-19"]
df_train.drop(["DATE"], axis=1, inplace=True)
df_train.drop(["SUM_COUNT"], axis=1, inplace=True)
df_test.drop(["DATE"], axis=1, inplace=True)
df_test.drop(["SUM_COUNT"], axis=1, inplace=True)

an_train = df_train["OUTLIER"]
an_test = df_test["OUTLIER"]

df_train.drop(["OUTLIER"], axis=1, inplace=True)
df_test.drop(["OUTLIER"], axis=1, inplace=True)


iso_forest = IsolationForest(contamination=0.06)
iso_forest.fit(df_train, an_train)

y_pred_iso = iso_forest.predict(df_test)  # predicted labels
y_pred_iso_map = np.where(y_pred_iso == -1, 1, 0)

print(classification_report(an_test, y_pred_iso_map))  # Isolation forest report

cm_iso = confusion_matrix_iso = pd.crosstab(
    an_test, y_pred_iso_map, rownames=["Actual"], colnames=["Predicted"]
)
print(confusion_matrix_iso)

df_final = df_test.copy()
df_final["OUTLIER"] = y_pred_iso_map
df_f_w = df_final[df_final["OUTLIER"] == 1]
f_appl = df_f_w["APPNAME"].unique()
f_apil = []
for i in f_appl:
    f_apil.append(df_f_w[df_f_w["APPNAME"] == i]["API"].unique())

final_list = pd.DataFrame()

for i, app_n in enumerate(f_appl):
    for j, api_n in enumerate(f_apil[i]):
        ind_df_f = pro_data[(pro_data["APPNAME"] == app_n) & (pro_data["API"] == api_n)]
        ind_df_f = ind_df_f[ind_df_f["DATE"] >= "2023-09-19"]
        ind_df_f.drop(["OUTLIER"], axis=1, inplace=True)
        ind_df_f["OUTLIER"] = df_final.loc[ind_df_f.index]["OUTLIER"]
        li = ind_df_f["APPNAME"].index

        if not (ind_df_f.OUTLIER.sum() == 0):
            if not (
                ind_df_f[ind_df_f["DATE"] == "2023-09-19"][
                    ind_df_f["OUTLIER"] == 1
                ].empty
            ):
                final_list = pd.concat(
                    [
                        final_list,
                        ind_df_f[ind_df_f["DATE"] == "2023-09-19"][
                            ind_df_f["OUTLIER"] == 1
                        ],
                    ],
                    ignore_index=True,
                )
                plt.figure(figsize=(10, 5))
                sns.scatterplot(
                    data=ind_df_f,
                    x="DATE",
                    y="SUM_COUNT",
                    hue=ind_df_f.OUTLIER.astype(bool),
                    palette=["green", "red"],
                )
                plt.title(
                    str(pro_df.loc[li[0]]["APPNAME"])
                    + " "
                    + str(pro_df.loc[li[0]]["API"])
                )
                plt.show()

print(final_list)
