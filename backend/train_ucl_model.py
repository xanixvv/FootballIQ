from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.ucl_team_names import normalize_ucl_team_name


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "ucl_matches.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = MODEL_DIR / "footballiq_ucl_model.joblib"
WINDOW = 5

FEATURES = [
    "home_points_avg", "home_goals_for_avg", "home_goals_against_avg",
    "home_win_rate", "away_points_avg", "away_goals_for_avg",
    "away_goals_against_avg", "away_win_rate", "home_matches_played",
    "away_matches_played", "home_advantage",
]
TARGET_ORDER = ["H", "D", "A"]


def points_for(result, is_home):
    if result == "D":
        return 1
    if is_home and result == "H":
        return 3
    if not is_home and result == "A":
        return 3
    return 0


def build_features(df):
    """Build pre-match rolling form; same-date matches never leak into each other."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["home_team"] = df["home_team"].map(normalize_ucl_team_name)
    df["away_team"] = df["away_team"].map(normalize_ucl_team_name)
    df = df.dropna(subset=["date", "home_team", "away_team", "result"])
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    history = defaultdict(lambda: deque(maxlen=WINDOW))
    rows = []

    for _, date_group in df.groupby("date", sort=True):
        for _, match in date_group.iterrows():
            home, away = match["home_team"], match["away_team"]
            hh, ah = list(history[home]), list(history[away])

            def average(items, key):
                return float(np.mean([item[key] for item in items])) if items else np.nan

            rows.append({
                "date": match["date"], "season": match["season"],
                "home_team": home, "away_team": away,
                "home_points_avg": average(hh, "points"),
                "home_goals_for_avg": average(hh, "gf"),
                "home_goals_against_avg": average(hh, "ga"),
                "home_win_rate": average(hh, "win"),
                "away_points_avg": average(ah, "points"),
                "away_goals_for_avg": average(ah, "gf"),
                "away_goals_against_avg": average(ah, "ga"),
                "away_win_rate": average(ah, "win"),
                "home_matches_played": len(hh),
                "away_matches_played": len(ah),
                "home_advantage": 1.0,
                "target": match["result"],
            })

        for _, match in date_group.iterrows():
            home, away, result = match["home_team"], match["away_team"], match["result"]
            hg, ag = int(match["home_goals"]), int(match["away_goals"])
            history[home].append({
                "points": points_for(result, True), "gf": hg, "ga": ag,
                "win": int(result == "H"),
            })
            history[away].append({
                "points": points_for(result, False), "gf": ag, "ga": hg,
                "win": int(result == "A"),
            })
    return pd.DataFrame(rows)


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"UCL dataset not found: {DATA_PATH}\nRun: python -m backend.prepare_ucl_data")
    matches = pd.read_csv(DATA_PATH)
    required = {"date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"UCL dataset is missing columns: {sorted(missing)}")
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce", utc=True)
    matches = matches.dropna(subset=["date", "home_team", "away_team", "result", "home_goals", "away_goals"])
    matches = matches[matches["result"].isin(TARGET_ORDER)].copy()
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
    matches = matches.dropna(subset=["home_goals", "away_goals"])
    data = build_features(matches)
    seasons = sorted(data["season"].dropna().unique())
    if len(seasons) < 3:
        raise ValueError("Need at least three seasons for chronological evaluation.")
    test_seasons = seasons[-2:]
    train = data[~data["season"].isin(test_seasons)].copy()
    test = data[data["season"].isin(test_seasons)].copy()
    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty.")

    model = make_model()
    model.fit(train[FEATURES], train["target"])
    predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])
    classes = list(model[-1].classes_)

    print("\n========== UCL MODEL EVALUATION ==========")
    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")
    print(f"Training seasons: {sorted(train['season'].unique())}")
    print(f"Test seasons: {test_seasons}")
    print(f"Features: {len(FEATURES)}")
    print(f"Accuracy: {accuracy_score(test['target'], predictions):.4f}")
    print(f"Macro F1: {f1_score(test['target'], predictions, average='macro', zero_division=0):.4f}")
    print(f"Log loss: {log_loss(test['target'], probabilities, labels=classes):.4f}")
    print("\nClassification report:")
    print(classification_report(test["target"], predictions, labels=TARGET_ORDER, zero_division=0))

    final_model = make_model()
    final_model.fit(data[FEATURES], data["target"])
    artifact = {
        "model": final_model, "features": FEATURES,
        "target_classes": list(final_model[-1].classes_),
        "training_rows": len(data), "training_seasons": seasons,
        "test_seasons": test_seasons, "window": WINDOW,
        "competition": "UEFA Champions League", "requires_odds": False,
        "team_name_normalization": "ucl_team_names.normalize_ucl_team_name.v1",
    }
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"\nSaved UCL model: {OUTPUT_PATH}")
    print("Evaluation used held-out seasons; deployment model was refitted on all available seasons.")


if __name__ == "__main__":
    main()
