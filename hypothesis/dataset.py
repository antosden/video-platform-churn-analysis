from pathlib import Path
import pandas as pd

PATH = Path(__file__).parent.parent / 'videostreaming_platform.csv'

def load_data():
    df = pd.read_csv(PATH)
    return df

def clean_data(df):
    df["city"] = df["city"].fillna("Unknown")
    df["avg_min_watch_daily"] = pd.to_numeric(df["avg_min_watch_daily"], errors="coerce")
    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
    return df

def get_clean_dataset():
    return clean_data(load_data())