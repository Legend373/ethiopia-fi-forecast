import pandas as pd

REQUIRED_COLUMNS = [
    "record_id",
    "record_type",
    "pillar",
    "indicator_code",
    "source_name",
    "source_url",
    "confidence"
]

def validate():
    df = pd.read_csv("data/raw/ethiopia_fi_unified_data.csv")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    print("\n=== Schema Check ===")
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required columns present ✔")

    print("\nNull counts:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

if __name__ == "__main__":
    validate()
