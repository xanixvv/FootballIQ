
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
# 1. LOAD DATA AND CREATE THE SAME CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Exclude rows without a usable target.
df = df.dropna(subset=["market_value_in_eur"]).copy()
df = df[df["market_value_in_eur"] >= 0].copy()

# Same cold-start definition as the previous experiment.
test = df[
    (df["date"] >= CUTOFF)
    & df["previous_value_eur"].isna()
].copy()

# Historical valuation rows before the cutoff are eligible for training.
train = df[df["date"] < CUTOFF].copy()

if train.empty or test.empty:
    raise ValueError(
        f"Empty dataset split: train={len(train)}, test={len(test)}"
    )

# ---------------------------------------------------------
# 2. DEFINE FEATURES
# ---------------------------------------------------------

# Model A: original features.
basic_numeric = [
    "age",
    "height_in_cm",
    "minutes_played_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
]

basic_categorical = [
    "position",
    "sub_position",
    "foot",
]

# Model B adds historical club and competition context.
club_features = [
    "historical_club_id",
    "competition_id",
]

# Model C adds the newly engineered performance features.
enhanced_numeric = [
    "appearances_365d",
    "appearances_90d",
    "minutes_played_90d",
    "goals_90d",
    "assists_90d",
    "goals_per90_365d",
    "assists_per90_365d",
    "recent_minutes_share",
]

# Check that the regenerated dataset contains every expected feature.
required_columns = (
    basic_numeric
    + basic_categorical
    + club_features
    + enhanced_numeric
    + ["date", "market_value_in_eur", "previous_value_eur"]
)

missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns:\n"
        + "\n".join(missing_columns)
        + "\nCheck the dataset-building scripts and regenerate the data."
    )

# Use exactly the same rows for every model.
X_train_basic = train[basic_numeric + basic_categorical]
X_test_basic = test[basic_numeric + basic_categorical]

X_train_club = train[
    basic_numeric + basic_categorical + club_features
]
X_test_club = test[
    basic_numeric + basic_categorical + club_features
]

X_train_enhanced = train[
    basic_numeric
    + enhanced_numeric
    + basic_categorical
    + club_features
]
X_test_enhanced = test[
    basic_numeric
    + enhanced_numeric
    + basic_categorical
    + club_features
]

y_train = np.log1p(train["market_value_in_eur"].astype(float))
y_test = test["market_value_in_eur"].astype(float)

# ---------------------------------------------------------
# 3. BUILD THE MODEL
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


# ---------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------

def evaluate(name, model, X, y):
    # Model predicts log(1 + market value).
    predictions = np.expm1(model.predict(X))
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y, predictions)
    median_ae = median_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print(f"\n--- {name} ---")
    print(f"MAE:                   €{mae:,.0f}")
    print(f"Median absolute error: €{median_ae:,.0f}")
    print(f"RMSE:                  €{rmse:,.0f}")
    print(f"R²:                    {r2:.4f}")

    return predictions, {
        "model": name,
        "MAE_eur": mae,
        "Median_AE_eur": median_ae,
        "RMSE_eur": rmse,
        "R2": r2,
    }


print("\n========== DATA SPLIT ==========")
print(f"Training rows: {len(train):,}")
print(f"Cold-start test rows: {len(test):,}")
print(f"Training cutoff: {CUTOFF.date()}")
print(
    f"Test dates: {test['date'].min().date()} "
    f"to {test['date'].max().date()}"
)
print(f"Unique cold-start players: {test['player_id'].nunique():,}")

# ---------------------------------------------------------
# 5. TRAIN AND COMPARE ALL THREE MODELS
# ---------------------------------------------------------

model_a = make_model(basic_numeric, basic_categorical)

model_b = make_model(
    basic_numeric,
    basic_categorical + club_features,
)

model_c = make_model(
    basic_numeric + enhanced_numeric,
    basic_categorical + club_features,
)

print("\nTraining Model A: original features...")
model_a.fit(X_train_basic, y_train)

print("Training Model B: original features + club + competition...")
model_b.fit(X_train_club, y_train)

print("Training Model C: club-aware + enhanced performance features...")
model_c.fit(X_train_enhanced, y_train)

pred_a, metrics_a = evaluate(
    "MODEL A — ORIGINAL FEATURES",
    model_a,
    X_test_basic,
    y_test,
)

pred_b, metrics_b = evaluate(
    "MODEL B — CLUB + COMPETITION",
    model_b,
    X_test_club,
    y_test,
)

pred_c, metrics_c = evaluate(
    "MODEL C — CLUB + COMPETITION + NEW PERFORMANCE FEATURES",
    model_c,
    X_test_enhanced,
    y_test,
)

# ---------------------------------------------------------
# 6. COMPARISON TABLE
# ---------------------------------------------------------

metrics = pd.DataFrame([metrics_a, metrics_b, metrics_c])

print("\n========== MODEL COMPARISON ==========")
print(metrics.to_string(index=False, formatters={
    "MAE_eur": lambda x: f"€{x:,.0f}",
    "Median_AE_eur": lambda x: f"€{x:,.0f}",
    "RMSE_eur": lambda x: f"€{x:,.0f}",
    "R2": lambda x: f"{x:.4f}",
}))

mae_a = metrics_a["MAE_eur"]
mae_b = metrics_b["MAE_eur"]
mae_c = metrics_c["MAE_eur"]

if mae_a > 0:
    print(f"\nMAE improvement, B vs A: {(mae_a - mae_b) / mae_a * 100:+.2f}%")

if mae_b > 0:
    print(f"MAE improvement, C vs B: {(mae_b - mae_c) / mae_b * 100:+.2f}%")

# ---------------------------------------------------------
# 7. SAVE METRICS AND ROW-LEVEL PREDICTIONS
# ---------------------------------------------------------

metrics_path = MODELS_DIR / "cold_start_three_model_metrics.csv"
metrics.to_csv(metrics_path, index=False)

comparison_columns = [
    "player_id",
    "date",
    "market_value_in_eur",
    "historical_club_id",
    "competition_id",
]

comparison = test[comparison_columns].copy()

comparison["model_a_prediction_eur"] = pred_a
comparison["model_b_prediction_eur"] = pred_b
comparison["model_c_prediction_eur"] = pred_c

for letter, prediction in [
    ("a", pred_a),
    ("b", pred_b),
    ("c", pred_c),
]:
    comparison[f"model_{letter}_absolute_error_eur"] = (
        comparison["market_value_in_eur"] - prediction
    ).abs()

comparison_path = MODELS_DIR / "cold_start_three_model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nSaved metrics to:")
print(metrics_path)

print("\nSaved row-level predictions to:")
print(comparison_path)

# ---------------------------------------------------------
# 8. CHECK HOW THE ENHANCED MODEL PERFORMS BY VALUE BAND
# ---------------------------------------------------------

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

band_summary = comparison.groupby(
    "actual_value_band", observed=False
).agg(
    players=("player_id", "count"),
    mean_actual_eur=("market_value_in_eur", "mean"),
    model_b_mae_eur=("model_b_absolute_error_eur", "mean"),
    model_c_mae_eur=("model_c_absolute_error_eur", "mean"),
)

print("\n========== MODEL B VS C BY ACTUAL VALUE BAND ==========")
print(band_summary.to_string(formatters={
    "mean_actual_eur": lambda x: f"€{x:,.0f}",
    "model_b_mae_eur": lambda x: f"€{x:,.0f}",
    "model_c_mae_eur": lambda x: f"€{x:,.0f}",
}))
