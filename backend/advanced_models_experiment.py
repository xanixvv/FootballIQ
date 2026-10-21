
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from xgboost import XGBClassifier

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

ODDS = ["B365H", "B365D", "B365A"]


def prepare_data(df):
    df = df.copy()

    inverse = 1.0 / df[ODDS].astype(float)
    total = inverse.sum(axis=1)

    df["market_home_prob"] = inverse["B365H"] / total
    df["market_draw_prob"] = inverse["B365D"] / total
    df["market_away_prob"] = inverse["B365A"] / total

    return df


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.9,
                reg_lambda=5,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def main():
    data = prepare_data(load_dataset())

    columns = FEATURES + [
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ]

    # XGBoost expects numeric class labels 0, 1, 2.
    # Use H=0, D=1, A=2 consistently for training.
    encode = {"H": 0, "D": 1, "A": 2}
    decode = np.array(["H", "D", "A"])

    results = []

    for fold in WALK_FORWARD_FOLDS:
        train = data[data["season"].isin(fold["train"])].copy()
        test_all = data[data["season"] == fold["test"]].copy()
        test = test_all.loc[valid_odds_mask(test_all)].copy()

        if train.empty or test.empty:
            raise ValueError(f"Empty data for {fold['test']}")

        y_train = train["target"].map(encode).to_numpy()
        y_test = test["target"].to_numpy()

        print(f"\nTEST SEASON: {fold['test']}")

        for name, model in get_models().items():
            model.fit(train[columns], y_train)

            # Read probabilities in the model's class order and convert
            # them explicitly to FootballIQ's H, D, A order.
            raw_probs = model.predict_proba(test[columns])

            # Map model probability columns into H, D, A order
            class_order = model[-1].classes_
            probs = np.zeros((len(test), 3), dtype=float)

            for index, class_id in enumerate(class_order):
                probs[:, int(class_id)] = raw_probs[:, index]

            # Normalize AFTER mapping to the three outcome columns
            row_sums = probs.sum(axis=1, keepdims=True)
            probs = probs / np.clip(row_sums, 1e-12, None)

            # Predictions use H, D, A order
            predictions = decode[probs.argmax(axis=1)]

            # sklearn log_loss expects A, D, H order here
            probs_for_loss = probs[:, [2, 1, 0]]

            print(
                f"{name:22} probability row-sum range: "
                f"{probs.sum(axis=1).min():.6f}–{probs.sum(axis=1).max():.6f}"
            )

            scores = {
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(
                    y_test,
                    predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "log_loss": log_loss(
                    y_test,
                    probs_for_loss,
                    labels=LOG_LOSS_LABELS,
                ),
            }

            results.append({
                "test_season": fold["test"],
                "model": name,
                **scores,
            })

            print(
                f"{name:24} "
                f"Accuracy={scores['accuracy']:.4f} | "
                f"Macro F1={scores['macro_f1']:.4f} | "
                f"Log loss={scores['log_loss']:.4f}"
            )

    results_df = pd.DataFrame(results)
    output = REPORTS / "advanced_models_experiment.csv"
    results_df.to_csv(output, index=False)

    print("\nMEAN RESULTS ACROSS TEST SEASONS")
    print(
        results_df.groupby("model")[
            ["accuracy", "macro_f1", "log_loss"]
        ].mean().sort_values("log_loss").round(4).to_string()
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
