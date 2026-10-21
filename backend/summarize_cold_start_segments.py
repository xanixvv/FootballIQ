
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT / "data" / "processed"
    / "player_market_value_with_history_and_club.csv"
)
PREDICTIONS_PATH = (
    ROOT / "models"
    / "cold_start_weighting_strengths_predictions.csv"
)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

test = df[
    (df["date"] >= pd.Timestamp("2023-03-31"))
    & df["previous_value_eur"].isna()
    & df["market_value_in_eur"].notna()
].copy()

key = ["player_id", "date"]
prediction_col = "d__no_weighting_prediction_eur"

test = test.merge(
    pred[key + [prediction_col]],
    on=key,
    how="inner",
    validate="one_to_one",
)

test["actual"] = test["market_value_in_eur"]
test["predicted"] = test[prediction_col]
test["absolute_error"] = (
    test["actual"] - test["predicted"]
).abs()
test["underpredicted"] = test["predicted"] < test["actual"]

test["age_group"] = pd.cut(
    test["age"],
    bins=[0, 18, 21, 25, 100],
    labels=["Under 18", "18–20", "21–24", "25+"],
    right=False,
)

def summarize(group):
    return pd.Series({
        "players": len(group),
        "MAE_eur": group["absolute_error"].mean(),
        "underprediction_pct": (
            group["underpredicted"].mean() * 100
        ),
        "actual_mean_eur": group["actual"].mean(),
        "predicted_mean_eur": group["predicted"].mean(),
    })

for column in ["age_group", "position"]:
    print(f"\n========== BY {column.upper()} ==========")
    result = (
        test.groupby(column, observed=True)
        .apply(summarize, include_groups=False)
        .sort_values("MAE_eur", ascending=False)
    )
    print(result.to_string(
        formatters={
            "MAE_eur": lambda x: f"€{x:,.0f}",
            "actual_mean_eur": lambda x: f"€{x:,.0f}",
            "predicted_mean_eur": lambda x: f"€{x:,.0f}",
            "underprediction_pct": lambda x: f"{x:.1f}%",
        }
    ))

print("\n========== YOUNG ATTACKERS ==========")
young_attackers = test[
    (test["age"] < 21)
    & (test["position"] == "Attack")
].sort_values("absolute_error", ascending=False)

print(f"Players: {len(young_attackers)}")
if len(young_attackers):
    print(
        young_attackers[
            ["player_id", "age", "actual", "predicted",
             "absolute_error", "underpredicted"]
        ].head(10).to_string(index=False)
    )
