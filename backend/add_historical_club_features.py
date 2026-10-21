
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

source_path = DATA / "player_market_value_with_history.csv"
valuations_path = RAW / "player_valuations.csv"

df = pd.read_csv(source_path)
valuations = pd.read_csv(valuations_path)

df["date"] = pd.to_datetime(df["date"])
valuations["date"] = pd.to_datetime(valuations["date"])

# Keep the historical club and competition recorded with each valuation.
club_features = valuations[
    [
        "player_id",
        "date",
        "current_club_id",
        "player_club_domestic_competition_id",
    ]
].copy()

# Resolve any duplicate player/date valuation records consistently.
club_features = (
    club_features.sort_values(["player_id", "date"])
    .drop_duplicates(["player_id", "date"], keep="last")
)

# Avoid accidental column collisions.
for col in [
    "current_club_id",
    "player_club_domestic_competition_id",
]:
    if col in df.columns:
        df = df.drop(columns=col)

result = df.merge(
    club_features,
    on=["player_id", "date"],
    how="left",
    validate="one_to_one",
)

result = result.rename(
    columns={
        "current_club_id": "historical_club_id",
        "player_club_domestic_competition_id": "competition_id",
    }
)

output_path = DATA / "player_market_value_with_history_and_club.csv"
result.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(f"Rows: {len(result):,}")
print(
    "Missing historical club:",
    f"{result['historical_club_id'].isna().sum():,}",
)
print(
    "Missing competition:",
    f"{result['competition_id'].isna().sum():,}",
)
print("\nMost common competitions:")
print(result["competition_id"].value_counts().head(15).to_string())
print("\nExample rows:")
print(
    result[
        [
            "player_id",
            "date",
            "historical_club_id",
            "competition_id",
            "market_value_in_eur",
        ]
    ].head(10).to_string(index=False)
)
