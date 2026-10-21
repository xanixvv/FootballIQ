
from pathlib import Path
import pandas as pd

from backend.ucl_team_names import normalize_ucl_team_name

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "ucl_matches.csv"


teams = [
    "Sabah FK",
    "Fenerbahçe SK",
    "PAE AEK",
    "Como 1907",
    "LASK Linz",
    "Real Betis Balompié",
    "Borussia Dortmund",
    "Manchester City FC",
    "Manchester United FC",
    "Sporting Clube de Portugal",
    "Villarreal CF",
    "Real Madrid CF",
    "AEK Athen (GRE)",
]


if not CSV.exists():
    print(f"ERROR: Dataset not found: {CSV}")
    raise SystemExit(1)

df = pd.read_csv(CSV)

required = {"home_team", "away_team"}
if not required.issubset(df.columns):
    print("ERROR: CSV is missing home_team or away_team.")
    raise SystemExit(1)

df["home_key"] = df["home_team"].map(normalize_ucl_team_name)
df["away_key"] = df["away_team"].map(normalize_ucl_team_name)

print(f"\nDataset rows: {len(df)}")
print("\n========== TEAM MATCHING ==========\n")

for team in teams:
    key = normalize_ucl_team_name(team)

    home = df[df["home_key"] == key]
    away = df[df["away_key"] == key]
    matches = len(home) + len(away)

    print(f"{team}")
    print(f"  Normalized: {key}")
    print(f"  Historical matches: {matches}")

    
    if matches == 0:
        words = [
            word for word in key.split()
            if len(word) >= 4 and word not in {
                "club", "sporting", "football"
            }
        ]

        candidates = set()

        for word in words:
            candidates.update(
                df.loc[
                    df["home_key"].str.contains(
                        word, na=False, regex=False
                    ),
                    "home_team",
                ].tolist()
            )
            candidates.update(
                df.loc[
                    df["away_key"].str.contains(
                        word, na=False, regex=False
                    ),
                    "away_team",
                ].tolist()
            )

        print(f"  Possible dataset names: {sorted(candidates)[:20]}")


print("========== DONE ==========")
