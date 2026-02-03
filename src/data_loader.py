import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))



def load_unified():
    return pd.read_csv("../data/raw/ethiopia_fi_unified_data.csv")

def load_reference():
    return pd.read_csv("../data/raw/reference_codes.csv")

def load_all():
    return load_unified(), load_reference()

if __name__ == "__main__":
    df, ref = load_all()
    print("Unified shape:", df.shape)
    print("Reference shape:", ref.shape)
