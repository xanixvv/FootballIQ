
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

TRAIN_FILES = [
    "E0_2021_22.csv",
    "E0_2022_23.csv",
    "E0_2023_24.csv",
    "E0_2024_25.csv",
]
TEST_FILE = "E0.csv"

FORM_FEATURES = [
    "home_points_avg",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_points_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "home_home_points_avg",
    "home_home_goals_for_avg",
    "home_home_goals_against_avg",
    "away_away_points_avg",
    "away_away_goals_for_avg",
    "away_away_goals_against_avg",
    "points_difference",
    "goal_difference_difference",
]

ODDS_FEATURES = ["AvgH", "AvgD", "AvgA"]


def summarize_form(history, venue=None):
    if venue is not None:
        history = [match for match in history if match[3] == venue]

    recent = history[-5:]

    if not recent:
        return 1.0, 1.3, 1.3

    points = sum(match[0] for match in recent) / len(recent)
    goals_for = sum(match[1] for match in recent) / len(recent)
    goals_against = sum(match[2] for match in recent) / len(recent)

    return points, goals_for, goals_against


def build_form_features(df):
    df = df.copy()
    df["DateParsed"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(
        subset=["DateParsed", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    )
    df = df.sort_values(["DateParsed", "Time"])

    history = {}
    records = []
    labels = []
    test_flags = []
    odds_records = []

    for _, day in df.groupby("DateParsed", sort=True):
        updates = []

        for _, match in day.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            hf = summarize_form(history.get(home, []))
            af = summarize_form(history.get(away, []))
            hhf = summarize_form(history.get(home, []), "home")
            aaf = summarize_form(history.get(away, []), "away")

            records.append({
                "home_points_avg": hf[0],
                "home_goals_for_avg": hf[1],
                "home_goals_against_avg": hf[2],
                "away_points_avg": af[0],
                "away_goals_for_avg": af[1],
                "away_goals_against_avg": af[2],
                "home_home_points_avg": hhf[0],
                "home_home_goals_for_avg": hhf[1],
                "home_home_goals_against_avg": hhf[2],
                "away_away_points_avg": aaf[0],
                "away_away_goals_for_avg": aaf[1],
                "away_away_goals_against_avg": aaf[2],
                "points_difference": hf[0] - af[0],
                "goal_difference_difference":
                    (hf[1] - hf[2]) - (af[1] - af[2]),
            })

            odds_records.append({
                column: pd.to_numeric(match.get(column), errors="coerce")
                for column in ODDS_FEATURES
            })

            labels.append(match["FTR"])
            test_flags.append(match["is_test"])

            hg = int(match["FTHG"])
            ag = int(match["FTAG"])

            if hg > ag:
                hp, ap = 3, 0
            elif hg < ag:
                hp, ap = 0, 3
            else:
                hp, ap = 1, 1

            updates.append((home, away, hp, ap, hg, ag))

        for home, away, hp, ap, hg, ag in updates:
            history.setdefault(home, []).append((hp, hg, ag, "home"))
            history.setdefault(away, []).append((ap, ag, hg, "away"))

    X_form = pd.DataFrame(records, columns=FORM_FEATURES)
    X_odds = pd.DataFrame(odds_records, columns=ODDS_FEATURES)
    y = pd.Series(labels, name="FTR")
    is_test = pd.Series(test_flags, name="is_test")

    return X_form, X_odds, y, is_test


def main():
    frames = []

    for filename in TRAIN_FILES:
        df = pd.read_csv(RAW_DIR / filename)
        df["is_test"] = False
        frames.append(df)
        print(f"Loaded {filename}: {len(df)} matches")

    test_df = pd.read_csv(RAW_DIR / TEST_FILE)
    test_df["is_test"] = True
    frames.append(test_df)
    print(f"Loaded {TEST_FILE}: {len(test_df)} matches")

    all_matches = pd.concat(frames, ignore_index=True)

    X_form, X_odds, y, is_test = build_form_features(all_matches)

    X_train_form = X_form.loc[~is_test]
    X_test_form = X_form.loc[is_test]
    X_train_odds = X_odds.loc[~is_test]
    X_test_odds = X_odds.loc[is_test]
    y_train = y.loc[~is_test]
    y_test = y.loc[is_test]

    # Keep the same test matches for every comparison.
    valid = X_test_odds.notna().all(axis=1)
    X_test_form = X_test_form.loc[valid]
    X_test_odds = X_test_odds.loc[valid]
    y_test = y_test.loc[valid]

    # Remove rows with missing odds from training the odds model.
    train_valid = X_train_odds.notna().all(axis=1)
    X_train_odds = X_train_odds.loc[train_valid]
    X_train_form_odds = X_train_form.loc[train_valid]
    y_train_odds = y_train.loc[train_valid]

    # Baseline: always predict the most common training outcome.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_form, y_train)
    baseline_pred = baseline.predict(X_test_form)

    # Model A: football form features only.
    form_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    form_model.fit(X_train_form, y_train)
    form_pred = form_model.predict(X_test_form)

    # Model B: form plus odds.
    combined_train = X_train_form_odds.join(X_train_odds)
    combined_test = X_test_form.join(X_test_odds)

    combined_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    )
    combined_model.fit(combined_train, y_train_odds)
    combined_pred = combined_model.predict(combined_test)


    
    # Logistic Regression: form features only.
    logistic_model = make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    logistic_model.fit(X_train_form, y_train)
    logistic_pred = logistic_model.predict(X_test_form)

    print(
        "Logistic Regression accuracy: "
        f"{accuracy_score(y_test, logistic_pred):.1%}"
    )


    # Odds-only benchmark.
    odds_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    )
    odds_model.fit(X_train_odds, y_train_odds)
    odds_pred = odds_model.predict(X_test_odds)

    print("\nFOOTBALLIQ MODEL COMPARISON")
    print("=" * 35)
    print(f"Common test matches: {len(y_test)}")
    print(f"Baseline accuracy: {accuracy_score(y_test, baseline_pred):.1%}")
    print(f"Form-only accuracy: {accuracy_score(y_test, form_pred):.1%}")
    print(f"Odds-only accuracy: {accuracy_score(y_test, odds_pred):.1%}")
    print(f"Form + odds accuracy: {accuracy_score(y_test, combined_pred):.1%}")
    
    # Evaluate the quality of predicted probabilities.
    class_order = list(combined_model.classes_)
    combined_probabilities = combined_model.predict_proba(combined_test)

    print(
        "Form + odds log loss: "
        f"{log_loss(y_test, combined_probabilities, labels=class_order):.4f}"
    )

    
    # Convert bookmaker odds into normalized probabilities.
    market_probabilities = pd.DataFrame({
        "H": 1 / X_test_odds["AvgH"],
        "D": 1 / X_test_odds["AvgD"],
        "A": 1 / X_test_odds["AvgA"],
    }, index=X_test_odds.index)

    market_probabilities = market_probabilities.div(
        market_probabilities.sum(axis=1), axis=0
    )

    # Compare market probabilities against our model.
    market_log_loss = log_loss(
        y_test,
        market_probabilities[["A", "D", "H"]],
        labels=["A", "D", "H"],
    )

    odds_model_log_loss = log_loss(
        y_test,
        odds_model.predict_proba(X_test_odds),
        labels=list(odds_model.classes_),
    )

    print(f"Bookmaker implied-probability log loss: {market_log_loss:.4f}")
    print(f"Odds-only ML log loss: {odds_model_log_loss:.4f}")

    
    
    # Exploratory blend: 90% market, 10% FootballIQ form model.
    class_order = ["A", "D", "H"]

    form_probabilities = pd.DataFrame(
        form_model.predict_proba(X_test_form),
        columns=form_model.classes_,
        index=X_test_form.index,
    ).reindex(columns=class_order)

    market_probabilities_ordered = market_probabilities[
        class_order
    ]

    blended_probabilities = (
        0.90 * market_probabilities_ordered
        + 0.10 * form_probabilities
    )

    blended_log_loss = log_loss(
        y_test,
        blended_probabilities,
        labels=class_order,
    )

    print(
        "90% market + 10% form log loss: "
        f"{blended_log_loss:.4f}"
    )




if __name__ == "__main__":
    main()
