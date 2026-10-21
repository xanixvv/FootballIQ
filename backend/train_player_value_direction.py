
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)
VALUATIONS_FILE = ROOT / "data" / "raw" / "player_valuations.csv"
OUTPUT_FILE = ROOT / "models" / "player_value_direction_predictions.csv"
MODEL_FILE = ROOT / "models" / "player_value_direction_model.joblib"

TEST_START = pd.Timestamp("2025-01-01")
MIN_DAYS = 90
MAX_DAYS = 120
STABILITY_THRESHOLD = 0.01  # 1% of previous valuation


def main():
    print("Loading valuation history and historical features...")

    features = pd.read_csv(FEATURES_FILE, low_memory=False)
    valuations = pd.read_csv(VALUATIONS_FILE, low_memory=False)

    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")

    valuations["market_value_in_eur"] = pd.to_numeric(
        valuations["market_value_in_eur"], errors="coerce"
    )

    valuations = valuations.dropna(
        subset=["player_id", "date", "market_value_in_eur"]
    )

    # One valuation per player/date, robust to duplicate source rows.
    valuations = (
        valuations.groupby(["player_id", "date"], as_index=False)
        ["market_value_in_eur"]
        .median()
        .sort_values(["player_id", "date"])
    )

    # Pair each valuation with the player's next recorded valuation.
    valuations["target_date"] = (
        valuations.groupby("player_id")["date"].shift(-1)
    )
    valuations["target_value_eur"] = (
        valuations.groupby("player_id")["market_value_in_eur"].shift(-1)
    )

    valuations["days_to_target"] = (
        valuations["target_date"] - valuations["date"]
    ).dt.days

    valuations = valuations[
        valuations["days_to_target"].between(MIN_DAYS, MAX_DAYS)
    ].copy()

    valuations = valuations.rename(
        columns={"market_value_in_eur": "starting_value_eur"}
    )

    # Only use feature snapshots that match the starting valuation date.
    df = valuations.merge(
        features,
        on=["player_id", "date"],
        how="inner",
        suffixes=("", "_feature"),
    )

    # The starting valuation must be known at prediction time.
    df = df[
        df["starting_value_eur"].notna()
        & (df["starting_value_eur"] > 0)
        & df["target_value_eur"].notna()
    ].copy()

    change = df["target_value_eur"] - df["starting_value_eur"]
    threshold = df["starting_value_eur"] * STABILITY_THRESHOLD

    df["direction"] = np.select(
        [change > threshold, change < -threshold],
        ["Increase", "Decrease"],
        default="Near unchanged",
    )

    # Features must be available on the starting valuation date.
    numeric_features = [
        "age",
        "height_in_cm",
        "previous_value_eur",
        "previous_peak_value_eur",
        "days_since_previous_valuation",
        "appearances_365d",
        "minutes_played_365d",
        "goals_365d",
        "assists_365d",
        "yellow_cards_365d",
        "red_cards_365d",
        "appearances_90d",
        "minutes_played_90d",
        "goals_90d",
        "assists_90d",
        "goals_per90_365d",
        "assists_per90_365d",
        "recent_minutes_share",
    ]

    categorical_features = ["position", "sub_position", "foot"]

    missing = [
        col for col in numeric_features + categorical_features
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Chronological split; training targets must also precede the test period.
    train = df[
        (df["date"] < TEST_START)
        & (df["target_date"] < TEST_START)
    ].copy()

    test = df[df["date"] >= TEST_START].copy()

    X_train = train[numeric_features + categorical_features]
    y_train = train["direction"]
    X_test = test[numeric_features + categorical_features]
    y_test = test["direction"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.9,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    print(f"\nTraining examples: {len(train):,}")
    print(f"Test examples: {len(test):,}")
    print(f"Unique test players: {test['player_id'].nunique():,}")
    print(f"Test starting dates: {test['date'].min().date()} to "
          f"{test['date'].max().date()}")
    print(f"Test target dates: {test['target_date'].min().date()} to "
          f"{test['target_date'].max().date()}")

    print("\nActual test class distribution:")
    print(y_test.value_counts())

    results = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", model),
            ]
        )

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        print(f"\n========== {name} ==========")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=["Decrease", "Increase", "Near unchanged"],
            zero_division=0,
        ))
        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=["Decrease", "Increase", "Near unchanged"],
        ))

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "predictions": predictions,
            "pipeline": pipeline,
        })

    # Save predictions from the model with the best macro F1.
    best = max(results, key=lambda result: result["macro_f1"])
    output = test[
        ["player_id", "date", "target_date", "starting_value_eur",
         "target_value_eur", "direction"]
    ].copy()

    output["predicted_direction"] = best["predictions"]
    output["selected_model"] = best["model"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSelected model by macro F1: {best['model']}")

    # Refit the selected model on all available labeled history
    # for future predictions. Test metrics above remain holdout metrics.
    X_all = df[numeric_features + categorical_features]
    y_all = df["direction"]

    print("\nRefitting selected model on all eligible historical examples...")
    best["pipeline"].fit(X_all, y_all)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best["pipeline"],
            "model_name": best["model"],
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "stability_threshold": STABILITY_THRESHOLD,
            "min_days": MIN_DAYS,
            "max_days": MAX_DAYS,
            "training_cutoff": str(df["target_date"].max().date()),
        },
        MODEL_FILE,
    )

    print(f"Saved test predictions to: {OUTPUT_FILE}")
    print(f"Saved trained model to: {MODEL_FILE}")
    print(f"Training examples used for final model: {len(df):,}")

    # Save the final deployment model
    import joblib

    MODEL_FILE = ROOT / "models" / "player_value_forecast_v2.joblib"

    print("\nRefitting final model on all eligible historical examples...")

    X_all = examples[NUM + CAT]
    y_all_log = np.log1p(
        examples["target_value_eur"].astype(float)
    )

    model.fit(X_all, y_all_log)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "numeric_features": NUM,
            "categorical_features": CAT,
            "target_transform": "log1p",
            "min_days": MIN_DAYS,
            "max_days": MAX_DAYS,
            "training_cutoff": str(
                examples["target_date"].max().date()
            ),
        },
        MODEL_FILE,
    )

    print(f"Saved final valuation model: {MODEL_FILE}")
    print(f"Training examples used: {len(examples):,}")
    print(
        f"Latest labeled target date: "
        f"{examples['target_date'].max().date()}"
    )


if __name__ == "__main__":
    main()
