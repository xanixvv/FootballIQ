
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
DATA = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA / "player_market_value_with_history_and_club.csv"
CUTOFF = pd.Timestamp("2023-03-31")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Original features
numeric_base = [
    "age",
    "height_in_cm",
    "minutes_played_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
]

categorical_base = [
    "position",
    "sub_position",
    "foot",
]

# Historical context
club_features = [
    "historical_club_id",
    "competition_id",
]

# Test feature groups independently.
feature_groups = {
    "90-day activity": [
        "appearances_90d",
        "minutes_played_90d",
        "recent_minutes_share",
    ],
    "90-day goals and assists": [
        "goals_90d",
        "assists_90d",
    ],
    "Per-90 rates": [
        "goals_per90_365d",
        "assists_per90_365d",
    ],
    "All new features": [
        "appearances_365d",
        "appearances_90d",
        "minutes_played_90d",
        "goals_90d",
        "assists_90d",
        "goals_per90_365d",
        "assists_per90_365d",
        "recent_minutes_share",
    ],
}


def make_model(numeric_features, categorical_features):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=10,
                        ),
                    ),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestRegressor(
                n_estimators=50,
                min_samples_leaf=5,
                max_features=0.9,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def evaluate(name, model, X_test, y_test):
    predictions = np.maximum(
        np.expm1(model.predict(X_test)),
        0,
    )

    return predictions, {
        "model": name,
        "MAE_eur": mean_absolute_error(y_test, predictions),
        "Median_AE_eur": median_absolute_error(y_test, predictions),
        "RMSE_eur": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions),
    }


# --------------------------------------------------
# Load and split data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.dropna(subset=["market_value_in_eur"]).copy()
df = df[df["market_value_in_eur"] >= 0].copy()

train = df[df["date"] < CUTOFF].copy()

test = df[
    (df["date"] >= CUTOFF)
    & df["previous_value_eur"].isna()
].copy()

if train.empty or test.empty:
    raise ValueError("Training or cold-start test dataset is empty.")

required = (
    numeric_base
    + categorical_base
    + club_features
    + sorted(set(sum(feature_groups.values(), [])))
    + ["previous_value_eur", "market_value_in_eur", "player_id", "date"]
)

missing = sorted(set(required) - set(df.columns))

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

print("========== DATA SPLIT ==========")
print(f"Training rows: {len(train):,}")
print(f"Cold-start test rows: {len(test):,}")
print(f"Unique test players: {test['player_id'].nunique():,}")
print(f"Training cutoff: {CUTOFF.date()}")
print(
    f"Test dates: {test['date'].min().date()} "
    f"to {test['date'].max().date()}"
)

y_train = np.log1p(train["market_value_in_eur"].astype(float))
y_test = test["market_value_in_eur"].astype(float)

# Every experiment uses the same base features and rows.
base_numeric = numeric_base
base_categorical = categorical_base + club_features

X_train_base = train[base_numeric + base_categorical]
X_test_base = test[base_numeric + base_categorical]

results = []
predictions_by_model = {}

# --------------------------------------------------
# Train reference Model B
# --------------------------------------------------

print("\nTraining reference Model B: club + competition...")

baseline = make_model(base_numeric, base_categorical)
baseline.fit(X_train_base, y_train)

baseline_predictions, baseline_metrics = evaluate(
    "Model B — club + competition",
    baseline,
    X_test_base,
    y_test,
)

results.append(baseline_metrics)
predictions_by_model["Model B"] = baseline_predictions

print("\nReference Model B:")
print(f"MAE: €{baseline_metrics['MAE_eur']:,.0f}")
print(f"Median AE: €{baseline_metrics['Median_AE_eur']:,.0f}")
print(f"RMSE: €{baseline_metrics['RMSE_eur']:,.0f}")
print(f"R²: {baseline_metrics['R2']:.4f}")

# --------------------------------------------------
# Test each feature group independently
# --------------------------------------------------

for group_name, extra_features in feature_groups.items():
    print(f"\nTraining experiment: {group_name}...")

    numeric_features = numeric_base + extra_features
    categorical_features = categorical_base + club_features

    X_train = train[numeric_features + categorical_features]
    X_test = test[numeric_features + categorical_features]

    model = make_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    predictions, metrics = evaluate(
        f"Model B + {group_name}",
        model,
        X_test,
        y_test,
    )

    results.append(metrics)
    predictions_by_model[group_name] = predictions

    print(f"MAE: €{metrics['MAE_eur']:,.0f}")
    print(f"Median AE: €{metrics['Median_AE_eur']:,.0f}")
    print(f"RMSE: €{metrics['RMSE_eur']:,.0f}")
    print(f"R²: {metrics['R2']:.4f}")

# --------------------------------------------------
# Compare results against reference Model B
# --------------------------------------------------

metrics_df = pd.DataFrame(results)
baseline_mae = baseline_metrics["MAE_eur"]

metrics_df["MAE_change_vs_B_pct"] = (
    (baseline_mae - metrics_df["MAE_eur"])
    / baseline_mae
    * 100
)

metrics_df["MAE_change_vs_B_pct"] = (
    metrics_df["MAE_change_vs_B_pct"].round(2)
)

metrics_path = MODELS_DIR / "cold_start_feature_group_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n========== FINAL COMPARISON ==========")
print(
    metrics_df[
        [
            "model",
            "MAE_eur",
            "Median_AE_eur",
            "RMSE_eur",
            "R2",
            "MAE_change_vs_B_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "MAE_eur": lambda x: f"€{x:,.0f}",
            "Median_AE_eur": lambda x: f"€{x:,.0f}",
            "RMSE_eur": lambda x: f"€{x:,.0f}",
            "R2": lambda x: f"{x:.4f}",
            "MAE_change_vs_B_pct": lambda x: f"{x:+.2f}%",
        },
    )
)

# Save predictions for player-by-player investigation.
comparison = test[
    [
        "player_id",
        "date",
        "market_value_in_eur",
        "historical_club_id",
        "competition_id",
    ]
].copy()

for model_name, predictions in predictions_by_model.items():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("—", "")
        .replace("-", "")
    )

    comparison[f"{safe_name}_prediction_eur"] = predictions
    comparison[f"{safe_name}_absolute_error_eur"] = (
        comparison["market_value_in_eur"] - predictions
    ).abs()

predictions_path = MODELS_DIR / "cold_start_feature_group_predictions.csv"
comparison.to_csv(predictions_path, index=False)

print(f"\nSaved metrics: {metrics_path}")
print(f"Saved predictions: {predictions_path}")

best = metrics_df.loc[metrics_df["MAE_eur"].idxmin()]

print("\nLowest-MAE experiment:")
print(f"{best['model']}")
print(f"MAE: €{best['MAE_eur']:,.0f}")
print(f"Change vs Model B: {best['MAE_change_vs_B_pct']:+.2f}%")
