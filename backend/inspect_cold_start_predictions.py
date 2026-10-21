
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "cold_start_club_comparison.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

# Calculate signed errors:
# Positive = model underpredicted; negative = model overpredicted.
df["club_signed_error_eur"] = (
    df["market_value_in_eur"] - df["club_prediction_eur"]
)

df["club_error_pct"] = (
    df["club_signed_error_eur"]
    / df["market_value_in_eur"].replace(0, np.nan)
) * 100

print("\n========== DATA CHECK ==========")
print(f"Rows: {len(df):,}")
print(f"Unique players: {df['player_id'].nunique():,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Missing club predictions: {df['club_prediction_eur'].isna().sum()}")
print(f"Missing actual values: {df['market_value_in_eur'].isna().sum()}")

print("\n========== OVERALL MODEL B ==========")
print(f"Actual mean: €{df['market_value_in_eur'].mean():,.0f}")
print(f"Predicted mean: €{df['club_prediction_eur'].mean():,.0f}")
print(
    "Actual median: "
    f"€{df['market_value_in_eur'].median():,.0f}"
)
print(
    "Predicted median: "
    f"€{df['club_prediction_eur'].median():,.0f}"
)
print(
    "Underprediction rate: "
    f"{(df['club_signed_error_eur'] > 0).mean() * 100:.1f}%"
)

print("\n========== 20 LARGEST UNDERPREDICTIONS ==========")
under = df.sort_values(
    "club_signed_error_eur", ascending=False
).head(20)

print(
    under[
        [
            "player_id",
            "date",
            "market_value_in_eur",
            "club_prediction_eur",
            "club_signed_error_eur",
            "historical_club_id",
            "competition_id",
        ]
    ].to_string(index=False)
)

print("\n========== 20 LARGEST OVERPREDICTIONS ==========")
over = df.sort_values(
    "club_signed_error_eur", ascending=True
).head(20)

print(
    over[
        [
            "player_id",
            "date",
            "market_value_in_eur",
            "club_prediction_eur",
            "club_signed_error_eur",
            "historical_club_id",
            "competition_id",
        ]
    ].to_string(index=False)
)

print("\n========== ERROR BY ACTUAL VALUE ==========")
bins = [-np.inf, 100_000, 500_000, 1_000_000,
        5_000_000, 10_000_000, np.inf]

labels = [
    "Up to €100K",
    "€100K–€500K",
    "€500K–€1M",
    "€1M–€5M",
    "€5M–€10M",
    "Above €10M",
]

df["actual_value_band"] = pd.cut(
    df["market_value_in_eur"],
    bins=bins,
    labels=labels,
    right=True,
)

summary = df.groupby(
    "actual_value_band", observed=False
).agg(
    players=("player_id", "size"),
    mean_actual=("market_value_in_eur", "mean"),
    mean_prediction=("club_prediction_eur", "mean"),
    MAE=("club_absolute_error_eur", "mean"),
)

print(summary.to_string())

print("\nInspection complete. Original comparison file was not modified.")
