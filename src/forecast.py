import pandas as pd
import numpy as np
from scipy import stats
import src.impact_model as im  # use your Task 3 module

# -----------------------
# Forecasting Utilities
# -----------------------

def prepare_timeseries(df, indicator_code):
    """
    Aggregate indicator by year.
    """
    ts_df = df[df["indicator_code"] == indicator_code].copy()
    ts_df["year"] = pd.to_datetime(ts_df["observation_date"], errors="coerce").dt.year
    ts = ts_df.groupby("year")["value_numeric"].mean()
    return ts

def linear_forecast(ts, forecast_years=[2025,2026,2027]):
    """
    Fit linear trend and forecast future values with confidence intervals.
    """
    ts_clean = ts.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_clean.index, ts_clean.values)
    
    forecast_index = np.array(forecast_years)
    forecast_values = intercept + slope * forecast_index
    se = std_err * np.sqrt(1 + (forecast_index - np.mean(ts_clean.index))**2 / np.sum((ts_clean.index - np.mean(ts_clean.index))**2))
    
    forecast_df = pd.DataFrame({
        "forecast": forecast_values,
        "ci_lower": forecast_values - 1.96*se,
        "ci_upper": forecast_values + 1.96*se
    }, index=forecast_index)
    
    return forecast_df

def event_augmented_forecast(ts, df, forecast_years=[2025,2026,2027]):
    """
    Apply event effects to time series and forecast future values.
    """
    joined = im.join_events_with_impacts(df)
    adjusted_ts = im.apply_event_effects(ts, joined)
    forecast_df = linear_forecast(adjusted_ts, forecast_years)
    return forecast_df

def scenario_forecast(base_forecast, multiplier=1.0):
    """
    Generate scenario forecasts scaling base forecast by multiplier.
    """
    df = base_forecast.copy()
    df["forecast"] = df["forecast"] * multiplier
    df["ci_lower"] = df["ci_lower"] * multiplier
    df["ci_upper"] = df["ci_upper"] * multiplier
    return df
