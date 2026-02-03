import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import eda, forecast, impact_model

sns.set(style="whitegrid")

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/ethiopia_fi_unified_data.csv")
    return df

df = load_data()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Ethiopia Financial Inclusion Dashboard")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Trends", "Forecasts", "Inclusion Projections"]
)

# -----------------------
# Overview Page
# -----------------------
if page == "Overview":
    st.title("Overview of Financial Inclusion Metrics")

    # Key metrics cards (placeholders)
    acc_current = df[df["indicator_code"] == "ACC_OWNERSHIP"]["value_numeric"].dropna().iloc[-1]
    usage_current = df[df["indicator_code"] == "USG_DIGITAL_PAYMENT"]["value_numeric"].dropna().iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric("Current Account Ownership (%)", f"{acc_current:.1f}")
    col2.metric("Current Digital Payment Usage (%)", f"{usage_current:.1f}")

    st.subheader("P2P / ATM Crossover Ratio")
    st.text("Placeholder for P2P/ATM crossover visualization")

    st.subheader("Growth Rate Highlights")
    st.text("Placeholder for recent growth rates of access and usage indicators")

# -----------------------
# Trends Page
# -----------------------
elif page == "Trends":
    st.title("Historical Trends")

    indicator_options = df["indicator_code"].unique().tolist()
    selected_indicator = st.selectbox("Select Indicator", indicator_options)

    ts = eda.temporal_coverage(df)  # placeholder for time series aggregation

    st.subheader(f"{selected_indicator} Trend")
    st.text("Placeholder for interactive time series plot")
    st.text("Add date range slider and channel comparison here")

# -----------------------
# Forecasts Page
# -----------------------
elif page == "Forecasts":
    st.title("Forecasts for Access and Usage")

    forecast_options = ["Account Ownership", "Digital Payment Usage"]
    selected_forecast = st.selectbox("Select Forecast", forecast_options)

    # Placeholder: Prepare historical series
    if selected_forecast == "Account Ownership":
        ts = forecast.prepare_timeseries(df, "ACC_OWNERSHIP")
    else:
        ts = forecast.prepare_timeseries(df, "USG_DIGITAL_PAYMENT")

    baseline_forecast = forecast.linear_forecast(ts)
    event_forecast = forecast.event_augmented_forecast(ts, df)

    st.subheader("Baseline Forecast")
    st.text("Placeholder for baseline forecast chart with CI")

    st.subheader("Event-Augmented Forecast")
    st.text("Placeholder for event-adjusted forecast chart with CI")

    st.subheader("Scenario Analysis")
    st.text("Placeholder for optimistic/base/pessimistic scenario plot")

# -----------------------
# Inclusion Projections Page
# -----------------------
elif page == "Inclusion Projections":
    st.title("Financial Inclusion Rate Projections")

    # Scenario selector
    scenario = st.selectbox("Select Scenario", ["Optimistic", "Base", "Pessimistic"])
    st.text(f"Showing {scenario} scenario projections for Access and Usage")

    st.subheader("Progress Toward 60% Account Ownership Target")
    st.text("Placeholder for progress bar or gauge chart visualization")

    st.subheader("Key Questions for Consortium")
    st.text("1. What drives financial inclusion in Ethiopia?\n"
            "2. How do product launches and policy changes affect outcomes?\n"
            "3. Projected access and usage rates for 2025-2027?")
