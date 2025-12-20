from pathlib import Path
from typing import Tuple, List

import pandas as pd


def _find_csv_in_parent_dir(filename: str = "videostreaming_platform.csv") -> Path:
    """
    Ищем CSV в директории на уровень выше относительно hypothesis/dataset.py.
    Гарантируется, что он там лежит.
    """
    here = Path(__file__).resolve()
    csv_path = here.parent.parent / filename  # project/<file>.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    return csv_path


def load_data(csv_filename: str = "videostreaming_platform.csv") -> pd.DataFrame:
    csv_path = _find_csv_in_parent_dir(csv_filename)
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # минимальная чистка под наши гипотезы
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown").astype(str)

    for col in ["avg_min_watch_daily", "churn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # user_id иногда нужен как счётчик
    if "user_id" in df.columns:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")

    return df


def build_city_table(df: pd.DataFrame) -> pd.DataFrame:
    if not {"city", "avg_min_watch_daily", "churn", "user_id"}.issubset(df.columns):
        missing = {"city", "avg_min_watch_daily", "churn", "user_id"} - set(df.columns)
        raise ValueError(f"Missing required columns for H4: {missing}")

    tbl = (
        df.groupby("city", as_index=False)
        .agg(
            avg_watch=("avg_min_watch_daily", "mean"),
            churn_rate=("churn", "mean"),
            users=("user_id", "count"),
        )
    )
    tbl["churn_rate_%"] = tbl["churn_rate"] * 100
    return tbl


def prepare_dataset(
    csv_filename: str = "videostreaming_platform.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int]:
    """
    Возвращаем:
    - df (очищенный)
    - city_tbl (агрегаты по городам)
    - all_cities
    - city_count
    """
    df = clean_data(load_data(csv_filename))
    city_tbl = build_city_table(df)
    all_cities = sorted(city_tbl["city"].astype(str).unique().tolist())
    city_count = int(city_tbl["city"].nunique())
    return df, city_tbl, all_cities, city_count