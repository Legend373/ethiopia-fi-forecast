import pandas as pd
import numpy as np

# -----------------------
# Load + Join Impact Data
# -----------------------

def join_events_with_impacts(df):
    """
    Join impact_link records with event metadata using category/time alignment.

    Args:
        df (pd.DataFrame): Full dataset including events and impact_links

    Returns:
        pd.DataFrame: merged dataframe with impact + event info
    """

    # Separate events and impacts
    events = df[df["record_type"] == "event"].copy()
    impacts = df[df["record_type"] == "impact_link"].copy()

    # Convert observation_date to years
    events["event_year"] = pd.to_datetime(events["observation_date"], errors="coerce").dt.year
    impacts["impact_year"] = pd.to_datetime(impacts["observation_date"], errors="coerce").dt.year

    # Merge by category (fallback if no explicit parent_id exists)
    joined = impacts.merge(
        events,
        on="category",
        suffixes=("_impact", "_event"),
        how="left"
    )

    return joined


# -----------------------
# Association Matrix
# -----------------------

def build_association_matrix(joined):
    """
    Build an Event → Indicator impact matrix.

    Rows: event category
    Columns: affected indicator (indicator_impact)
    Values: impact_estimate_event
    """
    # Ensure numeric
    joined["impact_estimate_event"] = pd.to_numeric(joined["impact_estimate_event"], errors="coerce").fillna(0)

    # Pivot table
    matrix = joined.pivot_table(
        index="category",
        columns="indicator_impact",
        values="impact_estimate_event",
        aggfunc="mean"
    )

    return matrix.fillna(0)


# -----------------------
# Event Effect Function
# -----------------------

def apply_event_effects(indicator_series, events, decay=0.8):
    """
    Apply event shocks to a time series with optional decay and lag.
    
    Args:
        indicator_series (pd.Series): time-indexed series of indicator values
        events (pd.DataFrame): joined dataframe with 'event_year' and 'impact_estimate'
        decay (float): annual decay factor of the event's effect

    Returns:
        pd.Series: adjusted indicator series
    """
    adjusted = indicator_series.copy()

    for _, row in events.iterrows():
        # Skip events with missing year
        if pd.isna(row.get("event_year")):
            continue

        year = int(row["event_year"])
        lag = row.get("lag_months", 0) or 0
        impact = row.get("impact_estimate_event", 0) or 0

        # Apply lag in years
        year += int(lag / 12)

        # Apply impact with decay over future years
        for t in range(year, adjusted.index.max() + 1):
            adjusted.loc[t] += impact * (decay ** (t - year))

    return adjusted


# -----------------------
# Validation
# -----------------------

def validate_against_history(actual, predicted):
    """
    Compare actual vs predicted indicator values.
    
    Returns a dataframe with error metrics.
    """
    comparison = pd.DataFrame({
        "actual": actual,
        "predicted": predicted
    })

    comparison["error"] = comparison["predicted"] - comparison["actual"]
    comparison["abs_error"] = comparison["error"].abs()

    return comparison
