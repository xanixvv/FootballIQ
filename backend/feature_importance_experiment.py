
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

from backend.compare_models import WALK_FORWARD_FOLDS, load_dataset, valid_odds_mask
from backend.form_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

ODDS = ["B365H", "B365D", "B365A"]


def prepare_data(df):
    df = df.copy()
    inverse = 1.0 / df[ODDS].astype(float)
    total = inverse.sum(axis=1)

    df["market_home_prob"] = inverse["B365H"] / total
    df["market_draw_prob"] = inverse["B365D"] / total
    df["market_away_prob"] = inverse["B365A"] / total

    return df


def main():
    data = prepare_data(load_dataset())
    columns = FEATURES + [
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ]

    results = []

    for fold in WALK_FORWARD_FOLDS:
        train = data[
            data["season"].isin(fold["train"])
        ].copy()

        test = data[
            data["season"] == fold["test"]
        ].copy()

        # Evaluate only matches with valid bookmaker odds
        train = train.loc[valid_odds_mask(train)].copy()
        test = test.loc[valid_odds_mask(test)].copy()

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )

        model.fit(train[columns], train["target"])

        importance = permutation_importance(
            model,
            test[columns],
            test["target"],
            scoring="f1_macro",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

        fold_results = pd.DataFrame({
            "feature": columns,
            "importance": importance.importances_mean,
            "std": importance.importances_std,
            "test_season": fold["test"],
        })

        results.append(fold_results)

        print(f"\nTEST SEASON: {fold['test']}")
        print(
            fold_results.sort_values(
                "importance", ascending=False
            ).head(10).to_string(index=False)
        )

    report = pd.concat(results, ignore_index=True)
    summary = (
        report.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    output = REPORTS / "feature_importance.csv"
    summary.to_csv(output)

    print("\nAVERAGE FEATURE IMPORTANCE ACROSS TEST SEASONS")
    print(summary.to_string())
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
