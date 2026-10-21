
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "models" / "player_market_value_test_predictions.csv"
df = pd.read_csv(path)

df = df.merge(
    pd.read_csv(
        ROOT / "data" / "processed" /
        "player_market_value_with_history.csv",
        usecols=["player_id", "date", "previous_value_eur"],
    ),
    on=["player_id", "date"],
    how="left",
    validate="one_to_one",
)

df = df.dropna(subset=["previous_value_eur"]).copy()

actual = df["market_value_in_eur"]
baseline = df["previous_value_eur"]
model = df["history_enhanced_prediction_eur"]

bins = [-np.inf, 500_000, 1_500_000, 5_000_000, 15_000_000, np.inf]
labels = [
    "Up to €500K",
    "€500K–€1.5M",
    "€1.5M–€5M",
    "€5M–€15M",
    "Above €15M",
]

df["value_segment"] = pd.cut(
    actual, bins=bins, labels=labels, right=True
)

print("SEGMENT PERFORMANCE (grouped by actual valuation)")
for segment, group in df.groupby("value_segment", observed=True):
    baseline_mae = mean_absolute_error(
        group["market_value_in_eur"], group["previous_value_eur"]
    )
    model_mae = mean_absolute_error(
        group["market_value_in_eur"],
        group["history_enhanced_prediction_eur"],
    )

    print(f"\n{segment} | rows: {len(group):,}")
    print(f"Last-known-value MAE: €{baseline_mae:,.0f}")
    print(f"Random Forest MAE:    €{model_mae:,.0f}")
    print(f"MAE improvement:      €{baseline_mae - model_mae:,.0f}")
