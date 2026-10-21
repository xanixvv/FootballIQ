
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

for filename in [
    "players.csv",
    "appearances.csv",
    "player_valuations.csv",
    "games.csv",
    "clubs.csv",
]:
    path = RAW / filename

    print(f"\n{'=' * 60}")
    print(f"{filename}")

    if not path.exists():
        print("File not found")
        continue

    df = pd.read_csv(path, nrows=5)
    print("Columns:")
    print(df.columns.tolist())
    print("\nSample:")
    print(df.head(2).to_string(index=False))
