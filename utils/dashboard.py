import pandas as pd
import plotly.express as px

LOG_FILE = "logs/emotion_history.csv"


def load_history():

    try:
        return pd.read_csv(LOG_FILE)
    except:
        return pd.DataFrame()


def emotion_bar_chart(df):

    counts = df["Emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]

    fig = px.bar(
        counts,
        x="Emotion",
        y="Count",
        color="Emotion",
        title="Emotion Distribution"
    )

    return fig


def emotion_pie_chart(df):

    fig = px.pie(
        df,
        names="Emotion",
        title="Emotion Distribution"
    )

    return fig


def confidence_chart(df):

    fig = px.line(
        df,
        y="Confidence",
        title="Prediction Confidence"
    )

    return fig
