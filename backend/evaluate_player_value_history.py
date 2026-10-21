
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "models" / "player_market_value_test_predictions.csv"
)
DATA_PATH = (
    ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)

pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
data = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Attach the historical value to the saved test predictions.
history = data[
    ["player_id", "date", "previous_value_eur"]
]

pred = pred.merge(
    history,
    on=["player_id", "date"],
    how="left",
    validate="one_to_one",
)

# Compare only rows where the previous valuation is available.
valid = pred.dropna(subset=["previous_value_eur"]).copy()

actual = valid["market_value_in_eur"]
last_value = valid["previous_value_eur"]
model_prediction = valid["history_enhanced_prediction_eur"]

def report(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"MAE:                   €{mean_absolute_error(y_true, y_pred):,.0f}")
    print(
        "Median absolute error: "
        f"€{median_absolute_error(y_true, y_pred):,.0f}"
    )
    print(
        "RMSE:                  "
        f"€{np.sqrt(mean_squared_error(y_true, y_pred)):,.0f}"
    )
    print(f"R²:                    {r2_score(y_true, y_pred):.4f}")

print("Test rows with previous valuation:", len(valid))
print("Total test rows:", len(pred))

report("LAST KNOWN VALUE BASELINE", actual, last_value)
report("HISTORY-ENHANCED RANDOM FOREST", actual, model_prediction)

print("\nAverage actual value: €{:,.0f}".format(actual.mean()))
print(
    "Average last-known value: €{:,.0f}".format(last_value.mean())
)
print(
    "Average model prediction: €{:,.0f}".format(model_prediction.mean())
)
