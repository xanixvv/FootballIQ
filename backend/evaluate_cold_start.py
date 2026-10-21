
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" /
    "player_market_value_with_history.csv"
)

TARGET = "market_value_in_eur"

NUMERIC_FEATURES = [
    "age",
    "height_in_cm",
    "minutes_played_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
]

CATEGORICAL_FEATURES = [
    "position",
    "sub_position",
    "foot",
]

HISTORY_FEATURES = [
    "previous_value_eur",
    "previous_peak_value_eur",
    "days_since_previous_valuation",
]


def make_model(numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def report(name, actual, predicted):
    print(f"\n--- {name} ---")
    print(f"MAE:                   €{mean_absolute_error(actual, predicted):,.0f}")
    print(
        "Median absolute error: "
        f"€{median_absolute_error(actual, predicted):,.0f}"
    )
    print(
        "RMSE:                  "
        f"€{np.sqrt(mean_squared_error(actual, predicted)):,.0f}"
    )
    print(f"R²:                    {r2_score(actual, predicted):.4f}")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["date", TARGET, "player_id"]).copy()

    # Match the previous chronological evaluation.
    cutoff = pd.Timestamp("2023-03-31")

    # Evaluate only rows with no prior valuation.
    cold = df[
        (df["date"] >= cutoff)
        & (df["previous_value_eur"].isna())
    ].copy()

    if cold.empty:
        print("No cold-start rows found in the test period.")
        return

    # A strict cold-start model: do not give it any valuation-history data.
    cold_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # For comparison, train a model on all pre-cutoff records using
    # the same non-history features, then evaluate cold-start rows.
    train = df[df["date"] < cutoff].copy()

    print("\n========== COLD-START TEST ==========")
    print("Training rows:", f"{len(train):,}")
    print("Cold-start test rows:", f"{len(cold):,}")
    print("Unique cold-start players:", f"{cold['player_id'].nunique():,}")
    print("Test starts:", cold["date"].min().date())
    print("Test ends:", cold["date"].max().date())

    model = make_model(NUMERIC_FEATURES)
    print("\nTraining cold-start model...")
    model.fit(train[cold_features], np.log1p(train[TARGET].astype(float)))

    predictions = np.maximum(
        np.expm1(model.predict(cold[cold_features])),
        0,
    )

    actual = cold[TARGET].astype(float)
    report("COLD-START RANDOM FOREST", actual, predictions)

    # Save results for later inspection.
    results = cold[
        [
            "player_id",
            "date",
            TARGET,
            "age",
            "position",
            "sub_position",
            "minutes_played_365d",
            "goals_365d",
            "assists_365d",
        ]
    ].copy()

    results["predicted_market_value_eur"] = predictions
    results["absolute_error_eur"] = (
        actual.to_numpy() - predictions
    ).__abs__()

    output = ROOT / "models" / "cold_start_test_predictions.csv"
    results.to_csv(output, index=False)

    print("\nSaved predictions to:")
    print(output)


if __name__ == "__main__":
    main()
