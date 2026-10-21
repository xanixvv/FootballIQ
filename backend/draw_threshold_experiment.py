
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

THRESHOLDS = np.round(np.arange(0.15, 0.451, 0.01), 2)


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def draw_threshold_predictions(probabilities, threshold):
    predictions = []

    for home_p, draw_p, away_p in probabilities:
        if draw_p >= threshold and draw_p >= max(home_p, away_p):
            predictions.append("D")
        else:
            predictions.append("H" if home_p >= away_p else "A")

    return np.asarray(predictions)


def draw_precision(y_true, predictions):
    predicted_draws = predictions == "D"

    if not predicted_draws.any():
        return 0.0

    return float(np.mean(y_true[predicted_draws] == "D"))


def draw_recall(y_true, predictions):
    actual_draws = y_true == "D"

    if not actual_draws.any():
        return 0.0

    return float(np.mean(predictions[actual_draws] == "D"))


def evaluate(y_true, predictions, probabilities):
    # Probability columns are H, D, A; sklearn expects A, D, H here.
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
        "draw_precision": draw_precision(y_true, predictions),
        "draw_recall": draw_recall(y_true, predictions),
        "log_loss": log_loss(
            y_true,
            probabilities[:, order],
            labels=LOG_LOSS_LABELS,
        ),
    }


def main():
    dataset = load_dataset()
    results = []

    print("\nFOOTBALLIQ — DRAW THRESHOLD EXPERIMENT")
    print("Thresholds are selected on validation data only.\n")

    for fold in WALK_FORWARD_FOLDS:
        test_season = fold["test"]
        train = dataset[
            dataset["season"].isin(fold["train"])
        ].copy()

        test_all = dataset[
            dataset["season"] == test_season
        ].copy()

        test = test_all.loc[valid_odds_mask(test_all)].copy()

        if train.empty or test.empty:
            raise ValueError(f"Empty train/test split: {test_season}")

        # Reserve the latest training season for threshold selection.
        validation_season = fold["train"][-1]
        fit_train = train[
            train["season"] != validation_season
        ].copy()
        validation = train[
            train["season"] == validation_season
        ].copy()

        # First fold has no earlier full season to validate against.
        # Use a chronological split within its training data instead.
        if fit_train.empty:
            ordered = train.sort_values("date").copy()
            split = int(len(ordered) * 0.8)
            fit_train = ordered.iloc[:split].copy()
            validation = ordered.iloc[split:].copy()
            validation_season = "Chronological training split"

        if fit_train.empty or validation.empty:
            raise ValueError(f"Could not create validation data for {test_season}")

        validation_model = make_model()
        validation_model.fit(
            fit_train[FEATURES],
            fit_train["target"],
        )

        validation_probs = aligned_probabilities(
            validation_model,
            validation[FEATURES],
        )
        y_validation = validation["target"].to_numpy()

        # Select threshold using validation macro F1.
        threshold_scores = []

        for threshold in THRESHOLDS:
            predictions = draw_threshold_predictions(
                validation_probs, threshold
            )
            score = f1_score(
                y_validation,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
            threshold_scores.append((score, threshold))

        best_score = max(score for score, _ in threshold_scores)
        tied_thresholds = [
            threshold
            for score, threshold in threshold_scores
            if np.isclose(score, best_score)
        ]
        best_threshold = min(
            tied_thresholds,
            key=lambda value: abs(value - 0.30),
        )

        # Refit on all permitted training seasons.
        final_model = make_model()
        final_model.fit(train[FEATURES], train["target"])

        test_probs = aligned_probabilities(
            final_model,
            test[FEATURES],
        )
        y_test = test["target"].to_numpy()

        default_predictions = np.asarray(LABELS)[
            test_probs.argmax(axis=1)
        ]
        tuned_predictions = draw_threshold_predictions(
            test_probs, best_threshold
        )

        for method, predictions in [
            ("Default argmax", default_predictions),
            ("Tuned draw threshold", tuned_predictions),
        ]:
            metrics = evaluate(y_test, predictions, test_probs)

            results.append({
                "test_season": test_season,
                "validation_season": validation_season,
                "method": method,
                "selected_draw_threshold": (
                    best_threshold
                    if method == "Tuned draw threshold"
                    else np.nan
                ),
                "validation_macro_f1": best_score,
                "test_matches": len(test),
                **metrics,
            })

        print(
            f"{test_season}: selected threshold = "
            f"{best_threshold:.2f} "
            f"(validation macro F1 = {best_score:.4f})"
        )

    results_df = pd.DataFrame(results)
    output = REPORTS / "draw_threshold_experiment.csv"
    results_df.to_csv(output, index=False)

    print("\nRESULTS — UNSEEN TEST SEASONS")
    print(
        results_df[
            [
                "test_season",
                "method",
                "selected_draw_threshold",
                "accuracy",
                "macro_f1",
                "draw_precision",
                "draw_recall",
                "log_loss",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
