
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
from sklearn.model_selection import GroupShuffleSplit
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
    "previous_value_eur",
    "previous_peak_value_eur",
    "days_since_previous_valuation",
]

CATEGORICAL_FEATURES = [
    "position",
    "sub_position",
    "foot",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


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

    # Keep the same future test period as the previous evaluation.
    cutoff = pd.Timestamp("2023-03-31")

    # Reserve a reproducible 20% of player identities.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    # GroupShuffleSplit returns row positions and keeps each player's
    # identity entirely within one group.
    _, heldout_positions = next(
        splitter.split(df, groups=df["player_id"])
    )

    heldout_players = set(
        df.iloc[heldout_positions]["player_id"].unique()
    )

    # Train only on non-held-out players and earlier dates.
    train = df[
        (df["date"] < cutoff)
        & (~df["player_id"].isin(heldout_players))
    ].copy()

    # Test only on held-out players in the future period.
    test = df[
        (df["date"] >= cutoff)
        & (df["player_id"].isin(heldout_players))
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty.")

    print("\n========== UNSEEN-PLAYER SPLIT ==========")
    print("Training rows:", f"{len(train):,}")
    print("Testing rows:", f"{len(test):,}")
    print("Training players:", f"{train['player_id'].nunique():,}")
    print("Held-out test players:", f"{test['player_id'].nunique():,}")
    print("Training ends before:", cutoff.date())
    print("Testing starts:", test["date"].min().date())

    # Confirm there is no overlap in player identities.
    overlap = set(train["player_id"]) & set(test["player_id"])
    print("Player overlap:", len(overlap))
    assert len(overlap) == 0, "Player identity leakage detected!"

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nTraining model on non-held-out players...")
    model.fit(train[FEATURES], np.log1p(y_train))

    predictions = np.maximum(
        np.expm1(model.predict(test[FEATURES])),
        0,
    )

    report("UNSEEN-PLAYER RANDOM FOREST", y_test, predictions)

    # The previous-value baseline is possible only when history exists.
    valid_history = test["previous_value_eur"].notna()

    print(
        "\nTest rows with previous valuation:",
        f"{valid_history.sum():,} / {len(test):,}",
    )

    if valid_history.any():
        report(
            "LAST-KNOWN-VALUE BASELINE",
            y_test[valid_history],
            test.loc[valid_history, "previous_value_eur"],
        )

        report(
            "RANDOM FOREST — SAME ROWS",
            y_test[valid_history],
            predictions[valid_history.to_numpy()],
        )

    # Save test predictions for later inspection.
    results = test[
        [
            "player_id",
            "date",
            TARGET,
            "previous_value_eur",
            "age",
            "position",
            "sub_position",
        ]
    ].copy()

    results["predicted_market_value_eur"] = predictions

    output = ROOT / "models" / "unseen_player_test_predictions.csv"
    results.to_csv(output, index=False)

    print("\nSaved test predictions to:")
    print(output)


if __name__ == "__main__":
    main()
