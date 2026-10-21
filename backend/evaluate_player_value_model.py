
import pandas as pd
import numpy as np

path = "models/player_market_value_test_predictions.csv"
df = pd.read_csv(path)

actual = df["market_value_in_eur"]
predicted = df["predicted_market_value_eur"]

df["error_eur"] = predicted - actual
df["absolute_error_eur"] = df["error_eur"].abs()

print("\n========== ERROR BY PLAYER VALUE ==========")

bins = [0, 500_000, 1_500_000, 5_000_000,
        15_000_000, float("inf")]

labels = [
    "Up to €500K",
    "€500K–€1.5M",
    "€1.5M–€5M",
    "€5M–€15M",
    "Above €15M",
]

df["value_group"] = pd.cut(
    actual, bins=bins, labels=labels, include_lowest=True
)

summary = df.groupby("value_group", observed=False).apply(
    lambda group: pd.Series({
        "players": len(group),
        "MAE_eur": group["absolute_error_eur"].mean(),
        "median_error_eur": group["absolute_error_eur"].median(),
        "mean_actual_eur": group["market_value_in_eur"].mean(),
        "mean_predicted_eur": group["predicted_market_value_eur"].mean(),
    }),
    include_groups=False,
)

print(summary.round(0).to_string())

print("\n========== LARGEST PREDICTION ERRORS ==========")
columns = [
    "player_id", "date", "position",
    "market_value_in_eur",
    "predicted_market_value_eur",
    "absolute_error_eur",
]

print(
    df.nlargest(10, "absolute_error_eur")[columns]
    .round(0)
    .to_string(index=False)
)
