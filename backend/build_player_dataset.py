
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

players = pd.read_csv(DATA_DIR / "players.csv")
appearances = pd.read_csv(DATA_DIR / "appearances.csv")
valuations = pd.read_csv(DATA_DIR / "player_valuations.csv")

# Parse dates
players["date_of_birth"] = pd.to_datetime(
    players["date_of_birth"], errors="coerce"
)
appearances["date"] = pd.to_datetime(
    appearances["date"], errors="coerce"
)
valuations["date"] = pd.to_datetime(
    valuations["date"], errors="coerce"
)

# Remove invalid rows
appearances = appearances.dropna(
    subset=["player_id", "date"]
).copy()

valuations = valuations.dropna(
    subset=["player_id", "date", "market_value_in_eur"]
).copy()

valuations = valuations[
    valuations["market_value_in_eur"] > 0
].copy()

# Convert performance columns to numeric
performance_stats = [
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
]

for col in performance_stats:
    appearances[col] = pd.to_numeric(
        appearances[col], errors="coerce"
    ).fillna(0)

# Count recorded appearance rows.
# This is a count of records, not necessarily unique matches.
appearances["appearances"] = 1

stats = ["appearances"] + performance_stats

# Aggregate records by player and date
daily = (
    appearances.groupby(["player_id", "date"], as_index=False)[stats]
    .sum()
    .sort_values(["player_id", "date"])
)

# Cumulative totals let us calculate historical windows efficiently.
cumulative_columns = ["cum_" + col for col in stats]

daily[cumulative_columns] = (
    daily.groupby("player_id")[stats].cumsum()
)


def attach_cumulative(source, queries, date_column, prefix):
    left = queries[
        ["row_id", "player_id", date_column]
    ].copy()

    left = left.rename(columns={date_column: "query_date"})
    left = left.sort_values(["query_date", "player_id"])

    right = source[
        ["player_id", "date"] + cumulative_columns
    ].sort_values(["date", "player_id"])

    merged = pd.merge_asof(
        left,
        right,
        left_on="query_date",
        right_on="date",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    )

    merged[cumulative_columns] = (
        merged[cumulative_columns].fillna(0)
    )

    merged = merged.drop(columns=["date", "query_date"])

    return merged.rename(
        columns={
            col: prefix + col
            for col in cumulative_columns
        }
    )


print("Preparing valuation snapshots...")

valuations = valuations.reset_index(drop=True)
valuations["row_id"] = np.arange(len(valuations))

valuations["cutoff_365"] = (
    valuations["date"] - pd.Timedelta(days=365)
)
valuations["cutoff_90"] = (
    valuations["date"] - pd.Timedelta(days=90)
)

before = attach_cumulative(
    daily, valuations, "date", "before_"
)
before_365 = attach_cumulative(
    daily, valuations, "cutoff_365", "old365_"
)
before_90 = attach_cumulative(
    daily, valuations, "cutoff_90", "old90_"
)

dataset = valuations.merge(
    before, on=["row_id", "player_id"], how="left"
)
dataset = dataset.merge(
    before_365, on=["row_id", "player_id"], how="left"
)
dataset = dataset.merge(
    before_90, on=["row_id", "player_id"], how="left"
)

# Calculate rolling 365-day and 90-day totals
for col in stats:
    dataset[col + "_365d"] = (
        dataset["before_cum_" + col]
        - dataset["old365_cum_" + col]
    ).clip(lower=0)

    dataset[col + "_90d"] = (
        dataset["before_cum_" + col]
        - dataset["old90_cum_" + col]
    ).clip(lower=0)

# Rates per 90 minutes; avoid division by zero
minutes_365 = dataset["minutes_played_365d"].replace(0, np.nan)

dataset["goals_per90_365d"] = (
    dataset["goals_365d"] / minutes_365 * 90
).fillna(0)

dataset["assists_per90_365d"] = (
    dataset["assists_365d"] / minutes_365 * 90
).fillna(0)

# Share of annual minutes recorded in the most recent 90 days.
# This is a rough recency indicator, not a form rating.
dataset["recent_minutes_share"] = (
    dataset["minutes_played_90d"]
    / dataset["minutes_played_365d"].replace(0, np.nan)
).fillna(0).clip(0, 1)

# Add profile features.
# Current/highest market value is deliberately excluded.
profile_columns = [
    "player_id",
    "date_of_birth",
    "position",
    "sub_position",
    "height_in_cm",
    "foot",
]

profiles = players[profile_columns].drop_duplicates("player_id")

dataset = dataset.merge(
    profiles, on="player_id", how="left"
)

dataset["age"] = (
    (dataset["date"] - dataset["date_of_birth"]).dt.days
    / 365.2425
)

# Basic data-quality filtering
dataset = dataset[
    dataset["age"].between(15, 45)
].copy()

dataset = dataset[
    dataset["minutes_played_365d"] > 0
].copy()

# Select features for modelling
keep = [
    "player_id",
    "date",
    "market_value_in_eur",
    "age",
    "position",
    "sub_position",
    "height_in_cm",
    "foot",
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

dataset = dataset[keep].sort_values("date")

output_path = OUTPUT_DIR / "player_market_value_dataset.csv"
dataset.to_csv(output_path, index=False)

print("\n========== DATASET BUILT ==========")
print(f"Rows: {len(dataset):,}")
print(f"Unique players: {dataset['player_id'].nunique():,}")
print(
    f"Date range: {dataset['date'].min()} "
    f"to {dataset['date'].max()}"
)
print(f"Saved to: {output_path}")

print("\nNew feature summary:")
print(
    dataset[
        [
            "appearances_365d",
            "minutes_played_90d",
            "goals_per90_365d",
            "assists_per90_365d",
            "recent_minutes_share",
        ]
    ].describe().round(3).to_string()
)

print("\nSample rows:")
print(dataset.head().to_string(index=False))
