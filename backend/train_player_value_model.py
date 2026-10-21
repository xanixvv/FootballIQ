
from pathlib import Path
import json

import joblib
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

ORIGINAL_DATA_PATH = (
    ROOT / "data" / "processed" / "player_market_value_dataset.csv"
)
HISTORY_DATA_PATH = (
    ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

ORIGINAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENHANCED_FEATURES = ORIGINAL_FEATURES + HISTORY_FEATURES


def build_model(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
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


def report_metrics(name, actual, predicted):
    metrics = {
        "mae_eur": float(mean_absolute_error(actual, predicted)),
        "median_absolute_error_eur": float(
            median_absolute_error(actual, predicted)
        ),
        "rmse_eur": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(r2_score(actual, predicted)),
    }

    print(f"\n--- {name} ---")
    print(f"MAE:                   €{metrics['mae_eur']:,.0f}")
    print(
        "Median absolute error: "
        f"€{metrics['median_absolute_error_eur']:,.0f}"
    )
    print(f"RMSE:                  €{metrics['rmse_eur']:,.0f}")
    print(f"R²:                    {metrics['r2']:.4f}")

    return metrics


def main():
    print("Loading datasets...")

    original = pd.read_csv(ORIGINAL_DATA_PATH, parse_dates=["date"])
    enhanced = pd.read_csv(HISTORY_DATA_PATH, parse_dates=["date"])

    # Match the history features to the original dataset by player and date.
    # This avoids relying on CSV row order.
    history_columns = [
        "player_id",
        "date",
        *HISTORY_FEATURES,
    ]

    df = original.merge(
        enhanced[history_columns],
        on=["player_id", "date"],
        how="left",
        validate="one_to_one",
    )

    df = df.dropna(subset=["date", TARGET]).copy()

    # Preserve the original chronological split for a fair comparison.
    dates = np.sort(df["date"].unique())
    split_index = int(len(dates) * 0.80)
    cutoff_date = pd.Timestamp(dates[split_index])

    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()

    print("\n========== TIME SPLIT ==========")
    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Training ends before:", cutoff_date.date())
    print("Testing starts:", test["date"].min().date())

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    baseline_prediction = np.full(len(test), y_train.median())
    baseline_metrics = report_metrics(
        "MEDIAN BASELINE", y_test, baseline_prediction
    )

    # Model A: original features.
    print("\nTraining Model A: original features...")
    original_model = build_model(
        NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    original_model.fit(
        train[ORIGINAL_FEATURES], np.log1p(y_train)
    )

    original_predictions = np.maximum(
        np.expm1(original_model.predict(test[ORIGINAL_FEATURES])),
        0,
    )
    original_metrics = report_metrics(
        "MODEL A — ORIGINAL FEATURES",
        y_test,
        original_predictions,
    )

    # Model B: original features plus historical valuation features.
    print("\nTraining Model B: valuation-history features...")
    enhanced_model = build_model(
        NUMERIC_FEATURES + HISTORY_FEATURES,
        CATEGORICAL_FEATURES,
    )
    enhanced_model.fit(
        train[ENHANCED_FEATURES], np.log1p(y_train)
    )

    enhanced_predictions = np.maximum(
        np.expm1(enhanced_model.predict(test[ENHANCED_FEATURES])),
        0,
    )
    enhanced_metrics = report_metrics(
        "MODEL B — HISTORY-ENHANCED FEATURES",
        y_test,
        enhanced_predictions,
    )

    print("\n========== COMPARISON ==========")
    print(
        "MAE change: "
        f"€{original_metrics['mae_eur'] - enhanced_metrics['mae_eur']:,.0f} "
        "(positive means improvement)"
    )
    print(
        "RMSE change: "
        f"€{original_metrics['rmse_eur'] - enhanced_metrics['rmse_eur']:,.0f} "
        "(positive means improvement)"
    )
    print(
        "R² change: "
        f"{enhanced_metrics['r2'] - original_metrics['r2']:+.4f}"
    )

    # Save the enhanced model as the main model.
    model_path = MODEL_DIR / "player_market_value_model.joblib"
    joblib.dump(enhanced_model, model_path)

    metadata = {
        "model_type": "RandomForestRegressor",
        "target": TARGET,
        "target_transform": "log1p",
        "features": ENHANCED_FEATURES,
        "numeric_features": NUMERIC_FEATURES + HISTORY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "training_start": str(train["date"].min().date()),
        "training_end": str(train["date"].max().date()),
        "testing_start": str(test["date"].min().date()),
        "testing_end": str(test["date"].max().date()),
        "split_cutoff": str(cutoff_date.date()),
        "metrics": {
            "median_baseline": baseline_metrics,
            "original_features": original_metrics,
            "history_enhanced": enhanced_metrics,
        },
    }

    metadata_path = MODEL_DIR / "player_market_value_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    results = test[
        ["player_id", "date", TARGET, "age", "position", "sub_position"]
    ].copy()

    results["original_prediction_eur"] = original_predictions
    results["history_enhanced_prediction_eur"] = enhanced_predictions
    results["absolute_error_eur"] = (
        results[TARGET] - results["history_enhanced_prediction_eur"]
    ).abs()

    results.to_csv(
        MODEL_DIR / "player_market_value_test_predictions.csv",
        index=False,
    )

    comparison_path = MODEL_DIR / "player_market_value_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "median_baseline": baseline_metrics,
            "original_features": original_metrics,
            "history_enhanced": enhanced_metrics,
        }, f, indent=2)

    print("\n========== MODEL SAVED ==========")
    print(model_path)
    print(metadata_path)
    print(comparison_path)
    print(
        MODEL_DIR / "player_market_value_test_predictions.csv"
    )


if __name__ == "__main__":
    main()
