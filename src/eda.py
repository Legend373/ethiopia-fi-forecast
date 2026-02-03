import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# -----------------------
# Dataset Overview
# -----------------------

def summarize_dataset(df):
    summary = {
        "record_type": df["record_type"].value_counts(),
        "pillar": df["pillar"].value_counts(),
        "source_type": df["source_type"].value_counts(),
        "confidence": df["confidence"].value_counts()
    }
    return summary


def temporal_coverage(df):
    df["year"] = pd.to_datetime(df["observation_date"], errors="coerce").dt.year
    coverage = pd.pivot_table(
        df,
        values="record_id",
        index="indicator_code",
        columns="year",
        aggfunc="count"
    )
    return coverage


def plot_temporal_coverage(coverage):
    plt.figure(figsize=(14,8))
    sns.heatmap(coverage.notnull(), cbar=False)
    plt.title("Indicator Coverage by Year")
    plt.show()


def sparse_indicators(df, n=10):
    return df.groupby("indicator_code").size().sort_values().head(n)


# -----------------------
# Access Analysis
# -----------------------

def account_trajectory(df):
    access = df[df["indicator_code"] == "ACC_OWNERSHIP"].copy()
    access["year"] = pd.to_datetime(access["observation_date"]).dt.year
    access = access.sort_values("year")
    return access


def plot_account_trajectory(access):
    plt.plot(access["year"], access["value_numeric"], marker="o")
    plt.title("Account Ownership Trajectory")
    plt.ylabel("% adults")
    plt.xlabel("Year")
    plt.show()


def growth_rates(access):
    access["growth_pp"] = access["value_numeric"].diff()
    return access


def plot_growth(access):
    plt.bar(access["year"], access["growth_pp"])
    plt.title("Growth Between Survey Years")
    plt.show()


# -----------------------
# Usage Analysis
# -----------------------

def mobile_money_trend(df):
    usage = df[df["indicator_code"] == "MOB_MONEY"].copy()
    usage["year"] = pd.to_datetime(usage["observation_date"]).dt.year
    return usage


def plot_mobile_trend(usage):
    plt.plot(usage["year"], usage["value_numeric"], marker="o")
    plt.title("Mobile Money Penetration")
    plt.show()


# -----------------------
# Infrastructure
# -----------------------

def infrastructure_subset(df):
    return df[df["pillar"] == "INFRASTRUCTURE"]


# -----------------------
# Correlation
# -----------------------

def correlation_matrix(df):
    df["year"] = pd.to_datetime(df["observation_date"], errors="coerce").dt.year
    pivot = df.pivot_table(
        index="year",
        columns="indicator_code",
        values="value_numeric"
    )
    return pivot.corr()


def plot_correlation(corr):
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Indicator Correlation Matrix")
    plt.show()


# -----------------------
# Events
# -----------------------

def extract_events(df):
    return df[df["record_type"] == "event"]


def overlay_events(access, events):
    plt.plot(access["year"], access["value_numeric"], marker="o")

    for y in pd.to_datetime(events["observation_date"], errors="coerce").dt.year:
        plt.axvline(y, linestyle="--", alpha=0.3)

    plt.title("Events Overlay on Ownership Trend")
    plt.show()
