
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "data" / "processed"
    / "player_market_value_with_history.csv"
)
VALUES_FILE = ROOT / "data" / "raw" / "player_valuations.csv"
OUTPUT_FILE = (
    ROOT / "models" / "player_value_forecast_v2_predictions.csv"
)

TEST_START = pd.Timestamp("2025-01-01")
MIN_DAYS = 90
MAX_DAYS = 120

NUM = [
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

CAT = ["position", "sub_position", "foot"]

print("Loading historical features and valuations...")

features = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
values = pd.read_csv(VALUES_FILE, parse_dates=["date"])

values = values.dropna(
    subset=["player_id", "date", "market_value_in_eur"]
)
values = values[values["market_value_in_eur"] >= 0].copy()

# One valuation per player/date, using the median if duplicates exist.
daily = (
    values.groupby(["player_id", "date"], as_index=False)
    ["market_value_in_eur"].median()
    .sort_values(["player_id", "date"])
    .reset_index(drop=True)
)

# Pair each valuation with the next recorded valuation for that player.
daily["target_value_eur"] = (
    daily.groupby("player_id")["market_value_in_eur"].shift(-1)
)
daily["target_date"] = daily.groupby("player_id")["date"].shift(-1)
daily["days_to_target"] = (
    daily["target_date"] - daily["date"]
).dt.days

targets = daily[
    daily["days_to_target"].between(MIN_DAYS, MAX_DAYS)
][
    ["player_id", "date", "target_date",
     "days_to_target", "target_value_eur"]
].copy()

# Attach features calculated strictly before each starting valuation date.
# These features include previous-value history and past performance.
examples = features.merge(
    targets,
    on=["player_id", "date"],
    how="inner",
    validate="one_to_one",
)

required = NUM + CAT + [
    "player_id", "date", "target_date",
    "target_value_eur", "previous_value_eur",
]
missing = sorted(set(required) - set(examples.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Use the carry-forward baseline, so require a previous recorded valuation.
examples = examples.dropna(
    subset=["previous_value_eur", "target_value_eur"]
).copy()

examples = examples[
    (examples["previous_value_eur"] >= 0)
    & (examples["target_value_eur"] >= 0)
].copy()

# Prevent training examples whose future target crosses into the test period.
train = examples[
    (examples["date"] < TEST_START)
    & (examples["target_date"] < TEST_START)
].copy()

test = examples[
    examples["date"] >= TEST_START
].copy()

if train.empty or test.empty:
    raise ValueError(
        f"Empty split: train={len(train)}, test={len(test)}"
    )

print(f"Training examples: {len(train):,}")
print(f"Testing examples:  {len(test):,}")
print(f"Unique test players: {test['player_id'].nunique():,}")
print(f"Training target dates end: {train['target_date'].max().date()}")
print(f"Test starting dates: {test['date'].min().date()} "
      f"to {test['date'].max().date()}")
print(f"Test target dates: {test['target_date'].min().date()} "
      f"to {test['target_date'].max().date()}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), NUM),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10,
        )),
    ]), CAT),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )),
])

X_train = train[NUM + CAT]
X_test = test[NUM + CAT]

y_train_log = np.log1p(
    train["target_value_eur"].astype(float)
)
actual = test["target_value_eur"].to_numpy(dtype=float)
baseline = test["previous_value_eur"].to_numpy(dtype=float)

print("\nTraining performance-feature model...")
model.fit(X_train, y_train_log)

predicted = np.maximum(
    np.expm1(model.predict(X_test)),
    0,
)

def report(label, truth, estimate):
    mae = mean_absolute_error(truth, estimate)
    rmse = np.sqrt(mean_squared_error(truth, estimate))
    print(f"\n{label}")
    print(f"  MAE:  €{mae:,.0f}")
    print(f"  RMSE: €{rmse:,.0f}")
    return mae, rmse

baseline_mae, baseline_rmse = report(
    "CARRY-FORWARD BASELINE", actual, baseline
)
model_mae, model_rmse = report(
    "PERFORMANCE-FEATURE MODEL", actual, predicted
)

print("\n========== COMPARISON ==========")
print(f"MAE improvement:  "
      f"{(1 - model_mae / baseline_mae) * 100:.2f}%")
print(f"RMSE improvement: "
      f"{(1 - model_rmse / baseline_rmse) * 100:.2f}%")

print("\nAverage actual value: €{:.0f}".format(actual.mean()))
print("Average model prediction: €{:.0f}".format(predicted.mean()))
print("Average baseline prediction: €{:.0f}".format(baseline.mean()))

result = test[
    [
        "player_id", "date", "target_date", "days_to_target",
        "previous_value_eur", "previous_peak_value_eur",
        "target_value_eur",
    ]
].copy()

result["predicted_value_eur"] = predicted
result["baseline_prediction_eur"] = baseline
result["absolute_error_eur"] = np.abs(actual - predicted)
result["baseline_absolute_error_eur"] = np.abs(actual - baseline)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved predictions: {OUTPUT_FILE}")
print(
    "\nReminder: the target is the next recorded valuation 90-120 days "
    "after the starting date, not an exact day-90 valuation."
)
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
print(f"Latest labeled target date: {examples['target_date'].max().date()}")
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
print(f"Latest labeled target date: {examples['target_date'].max().date()}")
