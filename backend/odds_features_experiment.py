
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.compare_models import (
    LABELS,
    LOG_LOSS_LABELS,
    WALK_FORWARD_FOLDS,
    load_dataset,
    valid_odds_mask,
    aligned_probabilities,
)
from backend.form_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


def add_odds_features(df):
    """Convert decimal odds into normalized implied probabilities."""
    result = df.copy()

    inverse_odds = 1.0 / result[ODDS_COLUMNS].astype(float)
    total = inverse_odds.sum(axis=1)

    result["market_home_prob"] = inverse_odds["B365H"] / total
    result["market_draw_prob"] = inverse_odds["B365D"] / total
    result["market_away_prob"] = inverse_odds["B365A"] / total

    return result


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def score_predictions(y_true, predictions, probabilities):
    order = [LABELS.index(label) for label in LOG_LOSS_LABELS]

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities[:, order],
            labels=LOG_LOSS_LABELS,
        ),
    }


def main():
    data = load_dataset()
    data = add_odds_features(data)

    # Compare form-only vs form + bookmaker probabilities on identical fixtures.
    feature_sets = {
        "Form only": FEATURES,
        "Form + bookmaker odds": FEATURES + [
            "market_home_prob",
            "market_draw_prob",
            "market_away_prob",
        ],
    }

    all_results = []

    for fold in WALK_FORWARD_FOLDS:
        train = data[data["season"].isin(fold["train"])].copy()
        test_all = data[data["season"] == fold["test"]].copy()
        test = test_all.loc[valid_odds_mask(test_all)].copy()

        if train.empty or test.empty:
            raise ValueError(f"Empty train/test split: {fold['test']}")

        print(f"\n{'=' * 65}")
        print(f"TEST SEASON: {fold['test']}")
        print(f"Training matches: {len(train)} | Test matches: {len(test)}")

        y_test = test["target"].to_numpy()

        for model_name, columns in feature_sets.items():
            model = make_model()
            model.fit(train[columns], train["target"])

            probabilities = aligned_probabilities(model, test[columns])
            predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]

            scores = score_predictions(y_test, predictions, probabilities)

            row = {
                "test_season": fold["test"],
                "model": model_name,
                "train_matches": len(train),
                "test_matches": len(test),
                **scores,
            }
            all_results.append(row)

            print(
                f"{model_name:24} "
                f"Accuracy={scores['accuracy']:.4f} | "
                f"Macro F1={scores['macro_f1']:.4f} | "
                f"Log loss={scores['log_loss']:.4f}"
            )

    results = pd.DataFrame(all_results)
    output = REPORTS / "odds_features_experiment.csv"
    results.to_csv(output, index=False)

    print("\nFINAL RESULTS")
    print(results.round(4).to_string(index=False))

    print("\nMEAN ACROSS TEST SEASONS")
    print(
        results.groupby("model")[
            ["accuracy", "macro_f1", "log_loss"]
        ].mean().round(4).to_string()
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
