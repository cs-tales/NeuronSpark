import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"])
    df = df.copy()
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.weekday
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def load_merged(data_dir: Path, base_name: str) -> pd.DataFrame:
    base = pd.read_csv(data_dir / base_name)
    calendar = pd.read_csv(data_dir / "calendar.csv")
    weather = pd.read_csv(data_dir / "weather.csv")
    menu = pd.read_csv(data_dir / "menu.csv")

    df = base.merge(calendar, on="date", how="left")
    df = df.merge(weather, on=["date", "meal"], how="left")
    df = df.merge(menu, on=["date", "meal", "canteen_area"], how="left")
    df = add_date_features(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-in", type=str, default="model.joblib")
    parser.add_argument("--out", type=str, default="results.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    payload = joblib.load(args.model_in)
    pipeline = payload["pipeline"]
    categorical_cols = payload["categorical_cols"]
    numeric_cols = payload["numeric_cols"]

    df = load_merged(data_dir, "test.csv")
    features = categorical_cols + numeric_cols
    preds = pipeline.predict(df[features])
    preds = np.clip(preds, 0.0, None)

    output = df[["date", "meal", "canteen_area"]].copy()
    output["volume"] = preds
    output.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
