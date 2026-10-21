
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VALUES_FILE = ROOT / "data" / "raw" / "player_valuations.csv"
DATASET_FILE = (
    ROOT / "data" / "processed" / "player_market_value_dataset.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)


def main():
    print("Loading datasets...")

    df = pd.read_csv(DATASET_FILE, parse_dates=["date"])
    values = pd.read_csv(VALUES_FILE, parse_dates=["date"])

    # Collapse multiple valuations for the same player on the same date.
    daily = (
        values.groupby(["player_id", "date"], as_index=False)
        ["market_value_in_eur"]
        .median()
        .sort_values(["date", "player_id"])
        .reset_index(drop=True)
    )

    # Keep the current valuation and the peak known by each historical date.
    daily["peak_to_date"] = (
        daily.groupby("player_id")["market_value_in_eur"].cummax()
    )

    history = daily[
        ["player_id", "date", "market_value_in_eur", "peak_to_date"]
    ].rename(
        columns={
            "date": "previous_valuation_date",
            "market_value_in_eur": "previous_value_eur",
            "peak_to_date": "previous_peak_value_eur",
        }
    )

    # merge_asof requires both tables to be sorted by date globally.
    targets = df.copy()
    targets["_row_id"] = range(len(targets))
    targets = targets.sort_values(["date", "player_id"])

    history = history.sort_values(
        ["previous_valuation_date", "player_id"]
    )

    result = pd.merge_asof(
        targets,
        history,
        left_on="date",
        right_on="previous_valuation_date",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    )

    # Optional, correctly calculated gap since the previous valuation.
    result["days_since_previous_valuation"] = (
        result["date"] - result["previous_valuation_date"]
    ).dt.days

    result = (
        result.sort_values("_row_id")
        .drop(columns="_row_id")
        .reset_index(drop=True)
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(result):,}")
    print(
        "Rows with previous valuation: "
        f"{result['previous_value_eur'].notna().sum():,}"
    )
    print(
        "Rows with previous peak: "
        f"{result['previous_peak_value_eur'].notna().sum():,}"
    )
    print(
        "Rows with missing previous valuation: "
        f"{result['previous_value_eur'].isna().sum():,}"
    )

    print("\nSample:")
    print(
        result[
            [
                "player_id",
                "date",
                "market_value_in_eur",
                "previous_value_eur",
                "previous_peak_value_eur",
                "days_since_previous_valuation",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
