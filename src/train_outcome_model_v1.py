
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
MODEL_PATH = PROJECT_DIR / "models" / "outcome_model.joblib"

TRAIN_FILES = [
    "E0_2021_22.csv",
    "E0_2022_23.csv",
    "E0_2023_24.csv",
    "E0_2024_25.csv",
]
TEST_FILE = "E0.csv"

FEATURES = [
    "home_points_avg",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_points_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
]


def summarize_form(history):
    recent = history[-5:]

    if not recent:
        return 1.0, 1.3, 1.3

    points = sum(match[0] for match in recent) / len(recent)
    goals_for = sum(match[1] for match in recent) / len(recent)
    goals_against = sum(match[2] for match in recent) / len(recent)

    return points, goals_for, goals_against


def build_features(df):
    df = df.copy()
    df["DateParsed"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["DateParsed"])
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df.sort_values(["DateParsed", "Time"])

    history = {}
    records = []
    labels = []
    test_flags = []

    # Process matches chronologically so future results never
    # contribute to earlier matches' form statistics.
    for _, day in df.groupby("DateParsed", sort=True):
        updates = []

        for _, match in day.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            home_form = summarize_form(history.get(home, []))
            away_form = summarize_form(history.get(away, []))

            records.append({
                "home_points_avg": home_form[0],
                "home_goals_for_avg": home_form[1],
                "home_goals_against_avg": home_form[2],
                "away_points_avg": away_form[0],
                "away_goals_for_avg": away_form[1],
                "away_goals_against_avg": away_form[2],
            })

            labels.append(match["FTR"])
            test_flags.append(match["is_test"])

            home_goals = int(match["FTHG"])
            away_goals = int(match["FTAG"])

            if home_goals > away_goals:
                home_points, away_points = 3, 0
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            # Save results for updating history AFTER today's
            # matches have had their features calculated.
            updates.append((
                home, away, home_points, away_points,
                home_goals, away_goals
            ))

        for home, away, hp, ap, hg, ag in updates:
            history.setdefault(home, []).append((hp, hg, ag))
            history.setdefault(away, []).append((ap, ag, hg))

    X = pd.DataFrame(records, columns=FEATURES)
    y = pd.Series(labels, name="FTR")
    is_test = pd.Series(test_flags, name="is_test")

    return X, y, is_test


def main():
    frames = []

    for filename in TRAIN_FILES:
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing training file: {path}")

        season = pd.read_csv(path)
        season["is_test"] = False
        frames.append(season)
        print(f"Loaded {filename}: {len(season)} matches")

    test_path = RAW_DIR / TEST_FILE

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    test_season = pd.read_csv(test_path)
    test_season["is_test"] = True
    frames.append(test_season)
    print(f"Loaded {TEST_FILE}: {len(test_season)} matches")

    all_matches = pd.concat(frames, ignore_index=True)
    X, y, is_test = build_features(all_matches)

    X_train = X.loc[~is_test]
    y_train = y.loc[~is_test]
    X_test = X.loc[is_test]
    y_test = y.loc[is_test]

    if X_train.empty or X_test.empty:
        raise ValueError("Training or test data is empty. Check your CSV files.")

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nFOOTBALLIQ - MULTI-SEASON PREDICTION MODEL")
    print("=" * 45)
    print(f"Total matches: {len(X)}")
    print(f"Training matches: {len(X_train)}")
    print(f"Test matches: {len(X_test)}")

    print(
        "\nBaseline accuracy: "
        f"{accuracy_score(y_test, baseline_predictions):.1%}"
    )
    print(
        "Model accuracy: "
        f"{accuracy_score(y_test, predictions):.1%}"
    )

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        target_names=["Home win", "Draw", "Away win"],
        zero_division=0,
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
