
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

players = pd.read_csv(RAW / "players.csv")
appearances = pd.read_csv(RAW / "appearances.csv", low_memory=False)
valuations = pd.read_csv(
    PROCESSED / "player_market_value_with_history_and_club.csv"
)

print("========== INTERNATIONAL DATA COVERAGE ==========")

for column in [
    "international_caps",
    "international_goals",
    "highest_market_value_in_eur",
]:
    if column in players.columns:
        print(f"\n{column}")
        print(f"Non-missing: {players[column].notna().sum():,}")
        print(f"Missing: {players[column].isna().sum():,}")
        print(f"Unique players with value: {players.loc[players[column].notna(), 'player_id'].nunique():,}")

print("\n========== HISTORICAL APPEARANCE DATA ==========")
print("Available appearance columns:")
print(appearances.columns.tolist())

print("\n========== COLD-START TEST ==========")
cutoff = pd.Timestamp("2023-03-31")
test = valuations[
    (pd.to_datetime(valuations["date"]) >= cutoff)
    & valuations["previous_value_eur"].isna()
].copy()

print(f"Cold-start rows: {len(test):,}")
print(f"Unique players: {test['player_id'].nunique():,}")

player_ids = set(test["player_id"].dropna())
international = players[players["player_id"].isin(player_ids)]

print(
    "Cold-start players with recorded international caps: "
    f"{international['international_caps'].notna().sum():,}"
)
print(
    "Cold-start players with recorded international goals: "
    f"{international['international_goals'].notna().sum():,}"
)

print(
    "\nImportant: these are current player-profile fields, not verified "
    "historical snapshots. Coverage alone does not make them safe "
    "for historical model training."
)
