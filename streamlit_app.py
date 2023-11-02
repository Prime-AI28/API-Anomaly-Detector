import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import src.Anomly_detection as ad
import src.data_preprocessing as dp
import src.Data_prediction as dpred

st.set_page_config(
    page_title="API MONITORING",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data


def slicer():
    default_app = "None"
    default_api = "None"

    dates = df_gp["DATE"].unique()

    app_options = df["APPNAME"].unique()
    app_options = ["None"] + [str(option) for option in app_options]

    st.write("Choose the APP and API to filter the data")
    Slicer_APP = st.selectbox("APPs", app_options, index=app_options.index(default_app))

    if Slicer_APP == "None":
        api_options = df["API"].unique()
    else:
        fil_app_df = df_gp[(df_gp["APPNAME"] == Slicer_APP)]
        api_options = fil_app_df["API"].unique()

    api_options = ["None"] + [str(option) for option in api_options]

    st.write("")
    Slicer_API = st.selectbox("APIs", api_options, index=api_options.index(default_api))

    if (Slicer_APP == "None") & (Slicer_API == "None"):
        filtered_df = t_data
    elif (Slicer_APP != "None") & (Slicer_API == "None"):
        filtered_df = df_gp[(df_gp["APPNAME"] == Slicer_APP)]
    elif (Slicer_APP == "None") & (Slicer_API != "None"):
        filtered_df = df_gp[(df_gp["API"] == Slicer_API)]
    else:
        filtered_df = fil_app_df[
            (fil_app_df["API"] == Slicer_API) & (fil_app_df["APPNAME"] == Slicer_APP)
        ]

    return filtered_df[-7:]


@st.cache_data
def plot_slicer(g_data):
    if len(g_data["API"].unique()) == 1:
        fig = px.line(
            g_data,
            x="DATE",
            y="SUM_COUNT",
            title=" API CALLS",
            color="APPNAME",
            markers=True,
            line_dash="API",
        )
    else:
        fig = px.line(
            g_data,
            x="DATE",
            y="SUM_COUNT",
            title=" API CALLS",
            color="API",
            markers=True,
            line_dash="APPNAME",
        )

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_base(g_data):
    fig = px.line(
        g_data,
        x="DATE",
        y="SUM_COUNT_DATE",
        title="API CALLS",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_simple(g_data, g_type, COUNT="COUNT"):
    trace1 = go.Scatter(
        x=g_data["DATE"],
        y=g_data[COUNT],
        mode="lines",
        line=dict(color="black"),
    )
    trace2 = go.Scatter(
        x=g_data["DATE"],
        y=g_data[COUNT],
        mode="markers",
        marker=dict(color=g_data["col"], size=5),
    )
    fig = go.Figure(
        data=[trace1, trace2], layout=go.Layout(title=go.layout.Title(text=g_type))
    )
    st.plotly_chart(fig)


@st.cache_data
def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={"prefix": prefix, "suffix": suffix, "font.size": 28},
            title={"text": label, "font": {"size": 24}},
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=(t_data["SUM_COUNT_DATE"]),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": indicator_suffix, "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 24},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=175,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def train(df):
    anomaly_list, message = dp.Anomaly_detection(df)
    st.write(message)
    return anomaly_list


@st.cache_data
def print_list():
    st.write("Anomaly Detected")
    st.dataframe(train(df))


def plot_pred(df):
    # Create a figure
    fig = px.line(x=df["ds"], y=df["y"])
    # Show the figure
    st.plotly_chart(fig, use_container_width=True)


st.title("API MONITORING")
st.markdown("_Prototype v0.1.1_")


with st.sidebar:
    st.header("Configuration")
    upload_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if upload_file is None:
    st.info("Please upload a file of type: " + ", ".join(["csv"]), icon="ℹ️")
    st.stop()

df = load_data(upload_file)
file = df.copy()
with st.expander("Data preview"):
    st.info("Showing data preview")
    st.markdown("### Raw data")
    st.dataframe(df)

col_1, col_2, col_3, col_4 = st.columns([1, 1, 1, 1])
col1, col2 = st.columns([4, 1])
b_l_c, b_r_c = st.columns(2)

pro_df, dates = dp.preprocessing(file)

dates.reverse()


# total data
t_data = df.copy()
t_data["SUM_COUNT_DATE"] = t_data.groupby(["DATE"])["COUNT"].transform("sum")
t_data = t_data.drop_duplicates(subset=["DATE"])
t_data = t_data.drop(["APPNAME", "API", "COUNT", "STATUS"], axis=1)
counts = sorted(t_data["SUM_COUNT_DATE"].tolist())
current_sumcount = t_data[t_data["DATE"] == dates[-1].strftime("%Y-%m-%d")][
    "SUM_COUNT_DATE"
].tolist()[0]
max_count = counts[-1]
per_total = (current_sumcount / max_count) * 100


# success plots
suc_data = df[(df["STATUS"] == "S")]
suc_data["SUM_COUNT_DATE"] = suc_data.groupby(["DATE"])["COUNT"].transform("sum")
suc_data = suc_data.drop_duplicates(subset=["DATE"])
suc_data = suc_data.drop(["APPNAME", "API", "COUNT"], axis=1)
suc_qr = np.percentile(suc_data["SUM_COUNT_DATE"], 5)
suc_nd = np.percentile(suc_data["SUM_COUNT_DATE"], 95)
suc_data["col"] = "blue"
suc_data.loc[suc_data["SUM_COUNT_DATE"] < suc_qr, "col"] = "red"

current_suc = suc_data[suc_data["DATE"] == dates[-1].strftime("%Y-%m-%d")][
    "SUM_COUNT_DATE"
].tolist()[0]
per_suc = (current_suc / suc_nd) * 100

# failiure plots
fail_data = df[(df["STATUS"] == "F")]
fail_data["SUM_COUNT_DATE"] = fail_data.groupby(["DATE"])["COUNT"].transform("sum")
fail_data = fail_data.drop_duplicates(subset=["DATE"])
fail_data = fail_data.drop(["APPNAME", "API", "COUNT"], axis=1)
fail_data["col"] = "blue"
fail_qr = np.percentile(fail_data["SUM_COUNT_DATE"], 90)
fail_data.loc[fail_data["SUM_COUNT_DATE"] > fail_qr, "col"] = "red"

current_fail = fail_data[fail_data["DATE"] == dates[-1].strftime("%Y-%m-%d")][
    "SUM_COUNT_DATE"
].tolist()[0]
per_fail = (current_fail / fail_qr) * 100


df_gp = df.copy()
df_gp["SUM_COUNT"] = df_gp.groupby(["DATE", "APPNAME", "API"])["COUNT"].transform("sum")
df_gp = df_gp.drop_duplicates(subset=["DATE", "APPNAME", "API"])
df_gp.drop(["COUNT", "STATUS"], axis=1, inplace=True)
df_gp = df_gp.reset_index(drop=True)


with col_1:
    plot_gauge(per_total, "#0068C9", "%", "Current Count Ratio", 100)
    plot_metric(
        label="D-1 API COUNT",
        value=current_sumcount,
        prefix="",
        suffix="",
        show_graph=True,
        color_graph="rgba(0, 104, 201, 0.2)",
    )
with col_2:
    plot_gauge(per_fail, "#E12800", "%", "Current Fail Ratio", 100)
    plot_metric(
        label="D-1 API Failure",
        value=current_fail,
        prefix="",
        suffix="",
        show_graph=True,
        color_graph="#ffcccb",
    )
with col_3:
    plot_gauge(per_suc, "#00E114", "%", "Current Success Ratio", 100)
    plot_metric(
        label="D-1 API Success",
        value=current_suc,
        prefix="",
        suffix="",
        show_graph=True,
        color_graph="#90EE90",
    )
with col_4:
    Date = st.selectbox("DATE", dates, index=dates.index(dates[0]))

    if st.button("Detect Anomalies"):
        an_list = ad.Anomalies(pro_df, Date)
        an_list.to_csv("an.csv", index=False)

    filter_data = slicer()
with col2:
    an_list = pd.read_csv("an.csv")
    custom_css = """
               <style>
               table {
                    font-size: 20px; /* Set your desired font size here */
               }
                    </style>
                    """
    # Display the custom CSS
    st.write(custom_css, unsafe_allow_html=True)
    #  Display the table with the custom CSS styling
    st.table(an_list)

with col1:
    if len(filter_data.columns.values) == 2:
        plot_base(filter_data)
    else:
        plot_slicer(filter_data)
with b_l_c:
    plot_simple(suc_data[-7:], "SUCCESS", "SUM_COUNT_DATE")
with b_r_c:
    plot_simple(fail_data[-7:], "FAIL", "SUM_COUNT_DATE")

unpro_df = df.copy()
clean_pro_df = dp.time_series_data(unpro_df)

if st.button("Predict"):
    future_data = dpred.prediction(clean_pro_df)
    plot_pred(future_data)
