
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

VALUES_FILE = ROOT / "data" / "raw" / "player_valuations.csv"
PLAYERS_FILE = ROOT / "data" / "raw" / "players.csv"
OUT_FILE = ROOT / "models" / "player_value_forecast_predictions.csv"

# Forecast target: next recorded valuation between 90 and 120 days later.
MIN_DAYS = 90
MAX_DAYS = 120
TEST_START = pd.Timestamp("2025-01-01")

# Load valuation history.
values = pd.read_csv(VALUES_FILE, parse_dates=["date"])
values = values.dropna(
    subset=["player_id", "date", "market_value_in_eur"]
)
values = values[values["market_value_in_eur"] >= 0].copy()

# Collapse to one valuation per player/date.
values = (
    values.groupby(["player_id", "date"], as_index=False)
    ["market_value_in_eur"].median()
    .sort_values(["player_id", "date"])
    .reset_index(drop=True)
)

# Starting value and next recorded valuation.
values["previous_value_eur"] = values["market_value_in_eur"]
values["target_value_eur"] = (
    values.groupby("player_id")["market_value_in_eur"].shift(-1)
)
values["target_date"] = values.groupby("player_id")["date"].shift(-1)
values["days_to_target"] = (
    values["target_date"] - values["date"]
).dt.days

# Only examples whose next valuation is 90-120 days later.
examples = values[
    values["days_to_target"].between(MIN_DAYS, MAX_DAYS)
].copy()

# Add profile features that are available independently of future valuations.
players = pd.read_csv(PLAYERS_FILE)

profile_cols = [
    "player_id",
    "date_of_birth",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
]
players = players[[c for c in profile_cols if c in players.columns]]
players = players.drop_duplicates("player_id", keep="first")

examples = examples.merge(
    players, on="player_id", how="left", validate="many_to_one"
)

# Age must be calculated as of the starting valuation date.
examples["date_of_birth"] = pd.to_datetime(
    examples["date_of_birth"], errors="coerce"
)
examples["age"] = (
    (examples["date"] - examples["date_of_birth"]).dt.days / 365.25
)

# Keep inputs that are known at the starting date.
NUM = [
    "previous_value_eur",
    "age",
    "height_in_cm",
    "days_to_target",
]
CAT = ["position", "sub_position", "foot"]

examples = examples.dropna(subset=["target_value_eur", "previous_value_eur"])
examples = examples[examples["age"].between(15, 45)].copy()

train = examples[examples["date"] < TEST_START].copy()
test = examples[examples["date"] >= TEST_START].copy()

if train.empty or test.empty:
    raise ValueError(
        f"Empty split: train={len(train)}, test={len(test)}. "
        "Check the data dates and TEST_START."
    )

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
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
y_train = np.log1p(train["target_value_eur"].astype(float))
y_test = test["target_value_eur"].astype(float)

model.fit(X_train, y_train)
predictions = np.maximum(
    np.expm1(model.predict(X_test)), 0
)

baseline = test["previous_value_eur"].to_numpy(dtype=float)
actual = y_test.to_numpy(dtype=float)

def report(label, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{label}:")
    print(f"  MAE:  €{mae:,.0f}")
    print(f"  RMSE: €{rmse:,.0f}")

print("========== FUTURE VALUATION FORECAST ==========")
print(f"Training examples: {len(train):,}")
print(f"Testing examples:  {len(test):,}")
print(f"Unique test players: {test['player_id'].nunique():,}")
print(f"Target horizon: {MIN_DAYS}-{MAX_DAYS} days")
print(f"Test start: {TEST_START.date()}")
print(f"Test target dates: {test['target_date'].min().date()} to "
      f"{test['target_date'].max().date()}")

report("CARRY-FORWARD BASELINE", actual, baseline)
report("RANDOM FOREST FORECAST", actual, predictions)

# Save individual predictions for further error analysis.
result = test[
    [
        "player_id", "date", "target_date",
        "days_to_target", "previous_value_eur", "target_value_eur",
    ]
].copy()
result["predicted_value_eur"] = predictions
result["baseline_prediction_eur"] = baseline
result["absolute_error_eur"] = np.abs(
    actual - predictions
)
result["baseline_absolute_error_eur"] = np.abs(
    actual - baseline
)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT_FILE, index=False)
print(f"\nSaved predictions: {OUT_FILE}")
