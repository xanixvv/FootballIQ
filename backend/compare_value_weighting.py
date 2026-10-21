
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

# ---------------------------------------------------------
# 1. LOAD DATA AND PRESERVE THE EXISTING COLD-START SPLIT
# ---------------------------------------------------------

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
    raise ValueError("Training or test dataset is empty.")

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
    "historical_club_id",
    "competition_id",
]

activity_features = [
    "appearances_90d",
    "minutes_played_90d",
    "recent_minutes_share",
]

required = (
    numeric_base
    + categorical_base
    + activity_features
    + ["date", "market_value_in_eur", "previous_value_eur", "player_id"]
)

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

# Keep the same feature columns for all three models.
base_features = numeric_base + categorical_base
activity_model_features = base_features + activity_features

X_train_base = train[base_features]
X_test_base = test[base_features]

X_train_activity = train[activity_model_features]
X_test_activity = test[activity_model_features]

y_train_eur = train["market_value_in_eur"].astype(float)
y_test_eur = test["market_value_in_eur"].astype(float)

y_train_log = np.log1p(y_train_eur)

print("========== DATA SPLIT ==========")
print(f"Training rows: {len(train):,}")
print(f"Cold-start test rows: {len(test):,}")
print(f"Unique test players: {test['player_id'].nunique():,}")
print(f"Training cutoff: {CUTOFF.date()}")
print(
    f"Test dates: {test['date'].min().date()} "
    f"to {test['date'].max().date()}"
)


# ---------------------------------------------------------
# 2. MODEL FACTORY
# ---------------------------------------------------------

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


def evaluate(name, model, X_test):
    predictions = np.maximum(
        np.expm1(model.predict(X_test)),
        0,
    )

    metrics = {
        "model": name,
        "MAE_eur": mean_absolute_error(y_test_eur, predictions),
        "Median_AE_eur": median_absolute_error(
            y_test_eur, predictions
        ),
        "RMSE_eur": np.sqrt(
            mean_squared_error(y_test_eur, predictions)
        ),
        "R2": r2_score(y_test_eur, predictions),
    }

    print(f"\n--- {name} ---")
    print(f"MAE: €{metrics['MAE_eur']:,.0f}")
    print(f"Median absolute error: €{metrics['Median_AE_eur']:,.0f}")
    print(f"RMSE: €{metrics['RMSE_eur']:,.0f}")
    print(f"R²: {metrics['R2']:.4f}")

    return predictions, metrics


# ---------------------------------------------------------
# 3. TRAIN REFERENCE MODEL B
# ---------------------------------------------------------

print("\nTraining Model B: club + competition...")

model_b = make_model(numeric_base, categorical_base)
model_b.fit(X_train_base, y_train_log)

pred_b, metrics_b = evaluate(
    "Model B — club + competition",
    model_b,
    X_test_base,
)


# ---------------------------------------------------------
# 4. TRAIN MODEL D: ADD 90-DAY ACTIVITY
# ---------------------------------------------------------

print("\nTraining Model D: Model B + 90-day activity...")

model_d = make_model(
    numeric_base + activity_features,
    categorical_base,
)
model_d.fit(X_train_activity, y_train_log)

pred_d, metrics_d = evaluate(
    "Model D — club + competition + 90-day activity",
    model_d,
    X_test_activity,
)


# ---------------------------------------------------------
# 5. TRAIN MODEL E: WEIGHT HIGHER-VALUE TRAINING ROWS
# ---------------------------------------------------------
# Weight tiers are determined ONLY from training targets.
# No test target is used to create training weights.
#
# Weight 1.0: lower 50% of training values
# Weight 1.5: 50th to 75th percentile
# Weight 2.0: 75th to 90th percentile
# Weight 3.0: top 10% of training values
#
# This is an experiment, not an assumption that these weights
# will improve the model.

q50, q75, q90 = y_train_eur.quantile([0.50, 0.75, 0.90])

sample_weights = np.select(
    [
        y_train_eur >= q90,
        y_train_eur >= q75,
        y_train_eur >= q50,
    ],
    [
        3.0,
        2.0,
        1.5,
    ],
    default=1.0,
)

print("\nTraining-value weighting thresholds:")
print(f"50th percentile: €{q50:,.0f} (weight 1.5 and above)")
print(f"75th percentile: €{q75:,.0f} (weight 2.0 and above)")
print(f"90th percentile: €{q90:,.0f} (weight 3.0)")

print("\nTraining Model E: activity features + value weighting...")

model_e = make_model(
    numeric_base + activity_features,
    categorical_base,
)

# Fit preprocessing on training data, then pass weights to the
# Random Forest estimator. This avoids relying on pipeline-specific
# sample-weight routing behavior.
preprocessing = model_e.named_steps["preprocessing"]
forest = model_e.named_steps["model"]

X_train_transformed = preprocessing.fit_transform(X_train_activity)
X_test_transformed = preprocessing.transform(X_test_activity)

forest.fit(
    X_train_transformed,
    y_train_log,
    sample_weight=sample_weights,
)

# Evaluate on the original, unweighted test cohort.
pred_e = np.maximum(
    np.expm1(forest.predict(X_test_transformed)),
    0,
)

metrics_e = {
    "model": "Model E — activity + value weighting",
    "MAE_eur": mean_absolute_error(y_test_eur, pred_e),
    "Median_AE_eur": median_absolute_error(y_test_eur, pred_e),
    "RMSE_eur": np.sqrt(mean_squared_error(y_test_eur, pred_e)),
    "R2": r2_score(y_test_eur, pred_e),
}

print(f"\n--- {metrics_e['model']} ---")
print(f"MAE: €{metrics_e['MAE_eur']:,.0f}")
print(f"Median absolute error: €{metrics_e['Median_AE_eur']:,.0f}")
print(f"RMSE: €{metrics_e['RMSE_eur']:,.0f}")
print(f"R²: {metrics_e['R2']:.4f}")


# ---------------------------------------------------------
# 6. OVERALL COMPARISON
# ---------------------------------------------------------

results = pd.DataFrame([metrics_b, metrics_d, metrics_e])
reference_mae = metrics_b["MAE_eur"]

results["MAE_improvement_vs_B_pct"] = (
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
            "MAE_improvement_vs_B_pct": lambda x: f"{x:+.2f}%",
        },
    )
)

metrics_path = MODELS_DIR / "cold_start_value_weighting_metrics.csv"
results.to_csv(metrics_path, index=False)


# ---------------------------------------------------------
# 7. COMPARE ERROR BY ACTUAL VALUE BAND
# ---------------------------------------------------------

comparison = test[
    ["player_id", "date", "market_value_in_eur"]
].copy()

comparison["model_b_prediction_eur"] = pred_b
comparison["model_d_prediction_eur"] = pred_d
comparison["model_e_prediction_eur"] = pred_e

for name in ["b", "d", "e"]:
    comparison[f"model_{name}_absolute_error_eur"] = (
        comparison["market_value_in_eur"]
        - comparison[f"model_{name}_prediction_eur"]
    ).abs()

comparison["actual_value_band"] = pd.cut(
    comparison["market_value_in_eur"],
    bins=[
        -1,
        100_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
        np.inf,
    ],
    labels=[
        "€0–100k",
        "€100–500k",
        "€500k–1m",
        "€1–5m",
        "€5–10m",
        "Over €10m",
    ],
)

band_results = comparison.groupby(
    "actual_value_band", observed=False
).agg(
    players=("player_id", "count"),
    model_b_mae=("model_b_absolute_error_eur", "mean"),
    model_d_mae=("model_d_absolute_error_eur", "mean"),
    model_e_mae=("model_e_absolute_error_eur", "mean"),
)

print("\n========== ERROR BY ACTUAL VALUE BAND ==========")
print(
    band_results.to_string(
        formatters={
            "model_b_mae": lambda x: f"€{x:,.0f}",
            "model_d_mae": lambda x: f"€{x:,.0f}",
            "model_e_mae": lambda x: f"€{x:,.0f}",
        },
    )
)

predictions_path = MODELS_DIR / "cold_start_value_weighting_predictions.csv"
comparison.to_csv(predictions_path, index=False)

print(f"\nSaved metrics: {metrics_path}")
print(f"Saved predictions: {predictions_path}")
