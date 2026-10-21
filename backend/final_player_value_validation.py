
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "player_market_value_with_history_and_club.csv"
OUT = ROOT / "models" / "final_player_value_validation.csv"

# Later chronological holdout; prior experiments used a March 2023 cutoff.
CUTOFF = pd.Timestamp("2025-01-01")

NUM = [
    "age", "height_in_cm", "minutes_played_365d",
    "goals_365d", "assists_365d", "yellow_cards_365d",
    "red_cards_365d", "appearances_90d",
    "minutes_played_90d", "recent_minutes_share",
]
CAT = [
    "position", "sub_position", "foot",
    "historical_club_id", "competition_id",
]

df = pd.read_csv(DATA, parse_dates=["date"])
df = df.dropna(subset=["market_value_in_eur"])
df = df[df["market_value_in_eur"] >= 0].copy()

# Only evaluate cold-start valuation records after cutoff.
train = df[df["date"] < CUTOFF].copy()
test = df[
    (df["date"] >= CUTOFF)
    & df["previous_value_eur"].isna()
].copy()

required = NUM + CAT + ["player_id", "market_value_in_eur"]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")
if train.empty or test.empty:
    raise ValueError("Training or test data is empty.")

X_train, X_test = train[NUM + CAT], test[NUM + CAT]
y_train = np.log1p(train["market_value_in_eur"].astype(float))
y_test = test["market_value_in_eur"].astype(float)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=10
        )),
    ]), CAT),
])

X_train_t = pre.fit_transform(X_train)
X_test_t = pre.transform(X_test)

model = RandomForestRegressor(
    n_estimators=50,
    min_samples_leaf=5,
    max_features=0.9,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_t, y_train)
pred = np.maximum(np.expm1(model.predict(X_test_t)), 0)

metrics = {
    "cutoff": CUTOFF.date(),
    "train_rows": len(train),
    "test_rows": len(test),
    "unique_test_players": test["player_id"].nunique(),
    "MAE_eur": mean_absolute_error(y_test, pred),
    "Median_AE_eur": np.median(np.abs(y_test - pred)),
    "RMSE_eur": np.sqrt(mean_squared_error(y_test, pred)),
    "R2": r2_score(y_test, pred),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([metrics]).to_csv(OUT, index=False)

print("========== CHRONOLOGICAL VALIDATION ==========")
for key, value in metrics.items():
    if key.endswith("_eur"):
        print(f"{key}: €{value:,.0f}")
    else:
        print(f"{key}: {value}")
print(f"\nSaved: {OUT}")
print(
    "\nCaution: this is a later historical holdout, but not a fully "
    "untouched evaluation because earlier experiments examined overlapping dates."
)
