
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from backend.compare_models import (
    WALK_FORWARD_FOLDS,
    load_dataset,
    valid_odds_mask,
)

from backend.form_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

ODDS = ["B365H", "B365D", "B365A"]
LABELS = ["H", "D", "A"]


def prepare_data(df):
    df = df.copy()
    inverse = 1.0 / df[ODDS].astype(float)
    total = inverse.sum(axis=1)

    df["market_home_prob"] = inverse["B365H"] / total
    df["market_draw_prob"] = inverse["B365D"] / total
    df["market_away_prob"] = inverse["B365A"] / total

    return df



def evaluate(y_true, probabilities):
    predictions = np.array(LABELS)[probabilities.argmax(axis=1)]

    # sklearn log_loss expects labels in lexicographic order: A, D, H.
    log_loss_order = ["A", "D", "H"]
    reordered_probabilities = probabilities[:, [2, 1, 0]]

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
            reordered_probabilities,
            labels=log_loss_order,
        ),
    }



def main():
    data = prepare_data(load_dataset())

    odds_features = [
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

        # Both models are evaluated on exactly the same fixtures.
        train = train.loc[valid_odds_mask(train)].copy()
        test = test.loc[valid_odds_mask(test)].copy()

        for model_name, columns in [
            ("Form only", FEATURES),
            ("Form + odds", FEATURES + odds_features),
        ]:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            )

            model.fit(train[columns], train["target"])

            raw = model.predict_proba(test[columns])
            model_classes = model[-1].classes_

            probabilities = np.zeros((len(test), 3))

            for i, label in enumerate(model_classes):
                probabilities[:, LABELS.index(label)] = raw[:, i]

            probabilities /= probabilities.sum(axis=1, keepdims=True)

            metrics = evaluate(
                test["target"].to_numpy(), probabilities
            )

            results.append({
                "test_season": fold["test"],
                "model": model_name,
                "matches": len(test),
                **metrics,
            })

            print(
                f"{fold['test']} | {model_name:12} | "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Macro F1={metrics['macro_f1']:.4f} | "
                f"Log loss={metrics['log_loss']:.4f}"
            )

    report = pd.DataFrame(results)
    report.to_csv(REPORTS / "final_model_experiment.csv", index=False)

    print("\nMEAN ACROSS TEST SEASONS")
    print(
        report.groupby("model")[
            ["accuracy", "macro_f1", "log_loss"]
        ].mean().sort_values("log_loss").to_string()
    )

    print(f"\nSaved: {REPORTS / 'final_model_experiment.csv'}")


if __name__ == "__main__":
    main()
