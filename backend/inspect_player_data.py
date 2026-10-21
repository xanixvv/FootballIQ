
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Load the two most important datasets
players = pd.read_csv(DATA_DIR / "players.csv")
appearances = pd.read_csv(DATA_DIR / "appearances.csv")
valuations = pd.read_csv(DATA_DIR / "player_valuations.csv")

print("\n========== DATASET SIZES ==========")
print("Players:", len(players))
print("Appearances:", len(appearances))
print("Valuations:", len(valuations))

print("\n========== DATE RANGES ==========")
for name, df in [
    ("Appearances", appearances),
    ("Valuations", valuations),
]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(
        name,
        "| Earliest:", df["date"].min(),
        "| Latest:", df["date"].max()
    )

print("\n========== MARKET VALUE QUALITY ==========")
print("Missing values:", valuations["market_value_in_eur"].isna().sum())
print("Zero values:", (valuations["market_value_in_eur"] == 0).sum())
print("Negative values:", (valuations["market_value_in_eur"] < 0).sum())
print("\nValue distribution (EUR):")
print(valuations["market_value_in_eur"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.99]
))

print("\n========== PLAYER COVERAGE ==========")
print("Unique players in profiles:", players["player_id"].nunique())
print("Unique players with appearances:", appearances["player_id"].nunique())
print("Unique players with valuations:", valuations["player_id"].nunique())

print("\n========== SAMPLE VALUATIONS ==========")
print(
    valuations.sort_values("date")
    [["player_id", "date", "market_value_in_eur", "current_club_name"]]
    .tail(10)
    .to_string(index=False)
)

print("\n========== DONE ==========")
