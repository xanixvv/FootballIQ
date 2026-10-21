
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

path = ROOT / "models" / "cold_start_test_predictions.csv"
df = pd.read_csv(path)

actual = df["market_value_in_eur"].astype(float)
predicted = df["predicted_market_value_eur"].astype(float)

# A simple baseline: predict the median value from the training period.
# This is the same training-period median used in the earlier evaluation.
training_data = pd.read_csv(
    ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)
training_data["date"] = pd.to_datetime(training_data["date"])

train = training_data[
    training_data["date"] < pd.Timestamp("2023-03-31")
]

baseline_value = train["market_value_in_eur"].median()
baseline = np.full(len(df), baseline_value)

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

print("Cold-start test rows:", len(df))
print(f"Training median baseline: €{baseline_value:,.0f}")
print(f"Mean actual value:        €{actual.mean():,.0f}")
print(f"Mean model prediction:    €{predicted.mean():,.0f}")

report("TRAINING-MEDIAN BASELINE", actual, baseline)
report("COLD-START RANDOM FOREST", actual, predicted)

print("\nLargest absolute errors:")
df["absolute_error_eur"] = (actual - predicted).abs()
print(
    df.nlargest(10, "absolute_error_eur")[
        [
            "player_id",
            "date",
            "market_value_in_eur",
            "predicted_market_value_eur",
            "absolute_error_eur",
        ]
    ].to_string(index=False)
)
