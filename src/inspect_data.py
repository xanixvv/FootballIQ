
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "data" / "raw" / "E0.csv"


def main():
    print("FOOTBALLIQ - MATCH OUTCOME ANALYSIS")
    print("=" * 40)

    # 1. Load the dataset.
    df = pd.read_csv(CSV_PATH)

    print(f"\nTotal matches: {len(df)}")

    # 2. Count each type of result.
    outcome_counts = df["FTR"].value_counts().reindex(
        ["H", "D", "A"], fill_value=0
    )

    outcome_names = {
        "H": "Home wins",
        "D": "Draws",
        "A": "Away wins",
    }

    print("\nMATCH OUTCOMES")
    for result_code, count in outcome_counts.items():
        percentage = count / len(df) * 100
        print(
            f"{outcome_names[result_code]}: "
            f"{count} ({percentage:.1f}%)"
        )

    # 3. Calculate average goals.
    average_home_goals = df["FTHG"].mean()
    average_away_goals = df["FTAG"].mean()

    print("\nGOAL STATISTICS")
    print(f"Average home goals: {average_home_goals:.2f}")
    print(f"Average away goals: {average_away_goals:.2f}")

    # 4. Find the most frequent scorelines.
    scorelines = (
        df.groupby(["FTHG", "FTAG"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\nTOP 5 MOST COMMON SCORELINES")
    for (home_goals, away_goals), count in scorelines.items():
        print(f"{home_goals}-{away_goals}: {count} matches")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
