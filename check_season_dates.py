
from pathlib import Path
import pandas as pd

raw = Path("data/raw")

for filename in [
    "E0_2021_22.csv",
    "E0_2022_23.csv",
    "E0_2023_24.csv",
    "E0_2024_25.csv",
    "E0.csv",
]:
    df = pd.read_csv(raw / filename)

    dates = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    ).dropna()

    print(f"\n{filename}")
    print("First 5 dates:")
    print(dates.sort_values().head().dt.strftime("%Y-%m-%d").to_list())
    print("Last 5 dates:")
    print(dates.sort_values().tail().dt.strftime("%Y-%m-%d").to_list())

    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        print("First fixture:")
        print(
            df.loc[dates.index[0], "HomeTeam"],
            "vs",
            df.loc[dates.index[0], "AwayTeam"],
        )
