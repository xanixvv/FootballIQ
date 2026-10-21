
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

numeric_features = [
    "age",
    "height_in_cm",
    "minutes_played_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
    "appearances_90d",
    "minutes_played_90d",
    "recent_minutes_share",
]

categorical_features = [
    "position",
    "sub_position",
    "foot",
    "historical_club_id",
    "competition_id",
]

# -------------------------------------------------------
# Load data and preserve the existing chronological split
# -------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.dropna(subset=["market_value_in_eur"]).copy()
df = df[df["market_value_in_eur"] >= 0].copy()

train = df[df["date"] < CUTOFF].copy()

test = df[
    (df["date"] >= CUTOFF)
    & df["previous_value_eur"].isna()
].copy()

required = (
    numeric_features
    + categorical_features
    + ["market_value_in_eur", "previous_value_eur", "player_id", "date"]
)

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if train.empty or test.empty:
    raise ValueError("Training or test dataset is empty.")

X_train = train[numeric_features + categorical_features]
X_test = test[numeric_features + categorical_features]

y_train_eur = train["market_value_in_eur"].astype(float)
y_train_log = np.log1p(y_train_eur)
y_test = test["market_value_in_eur"].astype(float)

print("========== DATA SPLIT ==========")
print(f"Training rows: {len(train):,}")
print(f"Cold-start test rows: {len(test):,}")
print(f"Unique test players: {test['player_id'].nunique():,}")
print(f"Training cutoff: {CUTOFF.date()}")

# -------------------------------------------------------
# Prepare the same preprocessing for every experiment
# -------------------------------------------------------

preprocessor = ColumnTransformer(
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
                ("imputer", SimpleImputer(strategy="most_frequent")),
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

print("\nPreprocessing features...")
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Training-only thresholds: do not use test targets for weights.
q50, q75, q90 = y_train_eur.quantile([0.50, 0.75, 0.90])

# Each tuple gives weights for:
# below Q50, Q50-Q75, Q75-Q90, and at/above Q90.
weighting_strategies = {
    "D — No weighting": (1.0, 1.0, 1.0, 1.0),
    "E1 — Gentle": (1.0, 1.25, 1.5, 2.0),
    "E2 — Moderate": (1.0, 1.5, 2.0, 2.5),
    "E3 — Strong": (1.0, 1.5, 2.0, 3.0),
}

print("\nTraining-value thresholds:")
print(f"Q50: €{q50:,.0f}")
print(f"Q75: €{q75:,.0f}")
print(f"Q90: €{q90:,.0f}")

# Define value bands once, using each row's training target.
training_bands = [
    y_train_eur < q50,
    (y_train_eur >= q50) & (y_train_eur < q75),
    (y_train_eur >= q75) & (y_train_eur < q90),
    y_train_eur >= q90,
]

# -------------------------------------------------------
# Train and evaluate each weighting strategy
# -------------------------------------------------------

all_metrics = []
all_predictions = {}

for strategy_name, weights_by_band in weighting_strategies.items():
    sample_weights = np.select(
        training_bands,
        weights_by_band,
        default=1.0,
    ).astype(float)

    model = RandomForestRegressor(
        n_estimators=50,
        min_samples_leaf=5,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )

    print(f"\nTraining {strategy_name}...")
    model.fit(
        X_train_encoded,
        y_train_log,
        sample_weight=sample_weights,
    )

    predictions = np.maximum(
        np.expm1(model.predict(X_test_encoded)),
        0,
    )

    all_predictions[strategy_name] = predictions

    metrics = {
        "strategy": strategy_name,
        "MAE_eur": mean_absolute_error(y_test, predictions),
        "Median_AE_eur": median_absolute_error(y_test, predictions),
        "RMSE_eur": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions),
    }

    all_metrics.append(metrics)

    print(f"MAE: €{metrics['MAE_eur']:,.0f}")
    print(f"Median AE: €{metrics['Median_AE_eur']:,.0f}")
    print(f"RMSE: €{metrics['RMSE_eur']:,.0f}")
    print(f"R²: {metrics['R2']:.4f}")

# -------------------------------------------------------
# Overall comparison
# -------------------------------------------------------

results = pd.DataFrame(all_metrics)
reference_mae = results.loc[
    results["strategy"] == "D — No weighting", "MAE_eur"
].iloc[0]

results["MAE_improvement_vs_D_pct"] = (
    (reference_mae - results["MAE_eur"])
    / reference_mae
    * 100
)

print("\n========== OVERALL COMPARISON ==========")
print(
    results.to_string(
        index=False,
        formatters={
            "MAE_eur": lambda x: f"€{x:,.0f}",
            "Median_AE_eur": lambda x: f"€{x:,.0f}",
            "RMSE_eur": lambda x: f"€{x:,.0f}",
            "R2": lambda x: f"{x:.4f}",
            "MAE_improvement_vs_D_pct": lambda x: f"{x:+.2f}%",
        },
    )
)

metrics_path = MODELS_DIR / "cold_start_weighting_strengths_metrics.csv"
results.to_csv(metrics_path, index=False)

# -------------------------------------------------------
# Error by actual market-value band
# -------------------------------------------------------

comparison = test[
    ["player_id", "date", "market_value_in_eur"]
].copy()

for strategy_name, predictions in all_predictions.items():
    safe_name = (
        strategy_name.lower()
        .replace(" ", "_")
        .replace("—", "")
        .replace("-", "")
    )

    comparison[f"{safe_name}_prediction_eur"] = predictions
    comparison[f"{safe_name}_abs_error_eur"] = (
        comparison["market_value_in_eur"] - predictions
    ).abs()

comparison["actual_value_band"] = pd.cut(
    comparison["market_value_in_eur"],
    bins=[-1, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, np.inf],
    labels=[
        "€0–100k",
        "€100–500k",
        "€500k–1m",
        "€1–5m",
        "€5–10m",
        "Over €10m",
    ],
)

error_columns = [
    col for col in comparison.columns
    if col.endswith("_abs_error_eur")
]

band_results = comparison.groupby(
    "actual_value_band", observed=False
)[error_columns].mean()

print("\n========== MEAN ABSOLUTE ERROR BY VALUE BAND ==========")
print(
    band_results.to_string(
        formatters={
            col: lambda x: f"€{x:,.0f}"
            for col in error_columns
        }
    )
)

predictions_path = (
    MODELS_DIR / "cold_start_weighting_strengths_predictions.csv"
)
comparison.to_csv(predictions_path, index=False)

print(f"\nSaved metrics: {metrics_path}")
print(f"Saved predictions: {predictions_path}")
