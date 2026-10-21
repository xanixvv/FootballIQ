
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

MODEL_PATH = MODELS / "player_market_value_model.joblib"
METADATA_PATH = MODELS / "player_market_value_metadata.json"
OUTPUT_PATH = PROCESSED / "current_player_value_predictions.csv"

# Use the day after the latest available appearance data.
# This is a historical-data cutoff, NOT today's live date.
CUTOFF_DATE = pd.Timestamp("2026-06-29")

FEATURES = [
    "age",
    "height_in_cm",
    "minutes_played_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
    "position",
    "sub_position",
    "foot",
    "previous_value_eur",
    "previous_peak_value_eur",
    "days_since_previous_valuation",
]


def main():
    print("Loading model and source data...")

    model = joblib.load(MODEL_PATH)

    import json
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("features") != FEATURES:
        raise ValueError(
            "Feature list differs from saved model metadata. "
            "Check player_market_value_metadata.json."
        )

    players = pd.read_csv(RAW / "players.csv")
    appearances = pd.read_csv(
        RAW / "appearances.csv",
        parse_dates=["date"],
    )
    valuations = pd.read_csv(
        RAW / "player_valuations.csv",
        parse_dates=["date"],
    )

    # Strictly exclude appearances on or after the cutoff.
    appearances = appearances[
        appearances["date"] < CUTOFF_DATE
    ].copy()

    # Keep only valid, positive historical valuations before cutoff.
    valuations = valuations.dropna(
        subset=["player_id", "date", "market_value_in_eur"]
    ).copy()
    valuations = valuations[
        (valuations["date"] < CUTOFF_DATE)
        & (valuations["market_value_in_eur"] > 0)
    ].copy()

    # Calculate the last 365 days of appearance statistics.
    window_start = CUTOFF_DATE - pd.Timedelta(days=365)
    recent = appearances[
        (appearances["date"] >= window_start)
        & (appearances["date"] < CUTOFF_DATE)
    ].copy()

    stat_columns = [
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
    ]

    for col in stat_columns:
        recent[col] = pd.to_numeric(
            recent[col], errors="coerce"
        ).fillna(0)

    stats = (
        recent.groupby("player_id", as_index=False)[stat_columns]
        .sum()
        .rename(columns={
            "minutes_played": "minutes_played_365d",
            "goals": "goals_365d",
            "assists": "assists_365d",
            "yellow_cards": "yellow_cards_365d",
            "red_cards": "red_cards_365d",
        })
    )

    # Attach the last known valuation and peak up to that date.
    valuations = valuations.sort_values(["player_id", "date"])
    valuations["peak_value"] = (
        valuations.groupby("player_id")["market_value_in_eur"].cummax()
    )

    latest_values = (
        valuations.groupby("player_id", as_index=False)
        .tail(1)[
            ["player_id", "date", "market_value_in_eur", "peak_value"]
        ]
        .rename(columns={
            "date": "previous_valuation_date",
            "market_value_in_eur": "previous_value_eur",
            "peak_value": "previous_peak_value_eur",
        })
    )

    # Profiles supply demographics and categories only.
    # Current market-value columns are deliberately excluded.
    profile_columns = [
        "player_id",
        "name",
        "date_of_birth",
        "position",
        "sub_position",
        "height_in_cm",
        "foot",
    ]
    profiles = players[profile_columns].drop_duplicates(
        "player_id", keep="last"
    ).copy()

    profiles["date_of_birth"] = pd.to_datetime(
        profiles["date_of_birth"], errors="coerce"
    )
    profiles["age"] = (
        (CUTOFF_DATE - profiles["date_of_birth"]).dt.days / 365.2425
    )

    # Only include players with usable profiles and prior valuations.
    dataset = profiles.merge(stats, on="player_id", how="left")
    dataset = dataset.merge(latest_values, on="player_id", how="inner")

    stat_features = [
        "minutes_played_365d",
        "goals_365d",
        "assists_365d",
        "yellow_cards_365d",
        "red_cards_365d",
    ]
    dataset[stat_features] = dataset[stat_features].fillna(0)

    dataset["days_since_previous_valuation"] = (
        CUTOFF_DATE - dataset["previous_valuation_date"]
    ).dt.days

    # Match the training dataset's basic eligibility rules.
    dataset = dataset[
        dataset["age"].between(15, 45)
        & (dataset["minutes_played_365d"] > 0)
    ].copy()

    # Preserve missing height/category values for the model's imputers.
    input_data = dataset[FEATURES].copy()

    print(f"Prediction cutoff: {CUTOFF_DATE.date()}")
    print(f"Eligible players: {len(input_data):,}")
    print("Generating predictions...")

    raw_predictions = np.asarray(model.predict(input_data)).reshape(-1)

    # Training target was log1p(market value in EUR).
    dataset["estimated_market_value_eur"] = np.maximum(
        0, np.expm1(raw_predictions)
    )

    dataset["prediction_cutoff_date"] = CUTOFF_DATE
    dataset["data_status"] = "historical_cutoff_estimate"

    output_columns = [
        "player_id",
        "name",
        "age",
        "position",
        "sub_position",
        "previous_valuation_date",
        "previous_value_eur",
        "previous_peak_value_eur",
        "days_since_previous_valuation",
        "minutes_played_365d",
        "goals_365d",
        "assists_365d",
        "estimated_market_value_eur",
        "prediction_cutoff_date",
        "data_status",
    ]

    dataset[output_columns].sort_values(
        "estimated_market_value_eur", ascending=False
    ).to_csv(OUTPUT_PATH, index=False)

    print("\n========== PREDICTIONS SAVED ==========")
    print(OUTPUT_PATH)
    print("\nTop 15 estimates:")
    print(
        dataset.sort_values(
            "estimated_market_value_eur", ascending=False
        )[[
            "name",
            "previous_value_eur",
            "estimated_market_value_eur",
            "previous_valuation_date",
        ]].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
