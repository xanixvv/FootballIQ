
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "player_market_value_with_history_and_club.csv"
)
PREDICTIONS_PATH = (
    ROOT / "models" / "cold_start_weighting_strengths_predictions.csv"
)
OUTPUT_PATH = ROOT / "models" / "cold_start_top_20_errors.csv"

CUTOFF = pd.Timestamp("2023-03-31")
TOP_N = 20


def main():
    df = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    df["date"] = pd.to_datetime(df["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    # Use exactly the same cold-start test cohort as the previous experiment.
    test = df[
        (df["date"] >= CUTOFF)
        & df["previous_value_eur"].isna()
        & df["market_value_in_eur"].notna()
    ].copy()

    # D is the unweighted baseline; it was best on overall MAE.
    prediction_column = "d__no_weighting_prediction_eur"

    if prediction_column not in predictions.columns:
        raise ValueError(
            f"Missing {prediction_column!r}. "
            f"Available columns: {predictions.columns.tolist()}"
        )

    # Join predictions to the source data using player ID and date.
    # Validate the join to avoid accidentally duplicating rows.
    key = ["player_id", "date"]

    if test.duplicated(key).any():
        raise ValueError(
            "The test dataset has duplicate player/date rows. "
            "We need to resolve the matching keys before joining."
        )

    if predictions.duplicated(key).any():
        raise ValueError(
            "The predictions file has duplicate player/date rows."
        )

    prediction_columns = key + [prediction_column]
    merged = test.merge(
        predictions[prediction_columns],
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(test):
        print(
            f"WARNING: matched {len(merged):,} of "
            f"{len(test):,} test rows."
        )

    merged = merged.rename(
        columns={prediction_column: "predicted_value_eur"}
    )

    merged["actual_value_eur"] = merged["market_value_in_eur"]
    merged["absolute_error_eur"] = (
        merged["actual_value_eur"] - merged["predicted_value_eur"]
    ).abs()

    merged["signed_error_eur"] = (
        merged["predicted_value_eur"] - merged["actual_value_eur"]
    )

    merged["error_direction"] = np.select(
        [
            merged["signed_error_eur"] < 0,
            merged["signed_error_eur"] > 0,
        ],
        [
            "Underpredicted",
            "Overpredicted",
        ],
        default="Exact",
    )

    # Show the model's prediction as a percentage of actual value.
    # This is undefined when actual value is zero.
    merged["prediction_to_actual_ratio"] = np.where(
        merged["actual_value_eur"] > 0,
        merged["predicted_value_eur"] / merged["actual_value_eur"],
        np.nan,
    )

    # Add readable value bands for diagnosis.
    merged["actual_value_band"] = pd.cut(
        merged["actual_value_eur"],
        bins=[-1, 100_000, 500_000, 1_000_000, 5_000_000,
              10_000_000, np.inf],
        labels=[
            "€0–100k",
            "€100–500k",
            "€500k–1m",
            "€1–5m",
            "€5–10m",
            "Over €10m",
        ],
    )

    # Prefer useful columns that actually exist in this dataset.
    preferred_columns = [
        "player_id",
        "date",
        "name",
        "player_name",
        "age",
        "position",
        "sub_position",
        "foot",
        "historical_club_id",
        "competition_id",
        "minutes_played_90d",
        "appearances_90d",
        "minutes_played_365d",
        "goals_365d",
        "assists_365d",
        "actual_value_eur",
        "predicted_value_eur",
        "signed_error_eur",
        "absolute_error_eur",
        "error_direction",
        "prediction_to_actual_ratio",
        "actual_value_band",
    ]

    output_columns = [
        col for col in preferred_columns if col in merged.columns
    ]

    top_errors = (
        merged.sort_values("absolute_error_eur", ascending=False)
        .head(TOP_N)[output_columns]
        .copy()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_errors.to_csv(OUTPUT_PATH, index=False)

    print("\n========== COLD-START ERROR INVESTIGATION ==========")
    print(f"Matched test rows: {len(merged):,}")
    print(f"Unique matched players: {merged['player_id'].nunique():,}")
    print(f"Rows exported: {len(top_errors):,}")

    print("\nUnderprediction vs overprediction across the test cohort:")
    print(merged["error_direction"].value_counts().to_string())

    print("\nLargest errors:")
    display_columns = [
        col for col in [
            "player_id",
            "date",
            "age",
            "position",
            "historical_club_id",
            "competition_id",
            "actual_value_eur",
            "predicted_value_eur",
            "absolute_error_eur",
            "error_direction",
        ]
        if col in top_errors.columns
    ]

    print(
        top_errors[display_columns].to_string(
            index=False,
            formatters={
                col: lambda x: f"€{x:,.0f}"
                for col in [
                    "actual_value_eur",
                    "predicted_value_eur",
                    "absolute_error_eur",
                ]
                if col in display_columns
            },
        )
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
