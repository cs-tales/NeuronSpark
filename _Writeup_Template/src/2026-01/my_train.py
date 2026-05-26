import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


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


def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_cols),
            ("numeric", numeric, numeric_cols),
        ]
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        l2_regularization=0.0,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-out", type=str, default="model.joblib")
    parser.add_argument("--val-days", type=int, default=60)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_merged(data_dir, "train.csv")

    categorical_cols = ["meal", "canteen_area", "weather", "menu_type"]
    numeric_cols = [
        "weekday",
        "is_weekend",
        "semester_week",
        "is_holiday",
        "is_exam_week",
        "is_makeup_day",
        "campus_event_level",
        "demand_index",
        "temperature",
        "feels_like",
        "humidity",
        "wind_speed",
        "rain_level",
        "menu_popularity",
        "is_promotion",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "sin_doy",
        "cos_doy",
    ]

    features = categorical_cols + numeric_cols
    df_features = df[features]
    target = df["volume"].astype(float)

    dates = pd.to_datetime(df["date"]).sort_values().unique()
    if args.val_days >= len(dates):
        raise ValueError("val-days is too large for the available dates")
    cutoff = dates[-args.val_days]
    is_val = pd.to_datetime(df["date"]) >= cutoff

    X_train = df_features.loc[~is_val]
    y_train = target.loc[~is_val]
    X_val = df_features.loc[is_val]
    y_val = target.loc[is_val]

    pipeline = build_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_pred = np.clip(val_pred, 0.0, None)

    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mape = np.mean(np.abs(y_val - val_pred) / np.maximum(y_val, 1e-6))

    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAPE: {mape:.4f}")

    joblib.dump(
        {
            "pipeline": pipeline,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
        },
        args.model_out,
    )
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
