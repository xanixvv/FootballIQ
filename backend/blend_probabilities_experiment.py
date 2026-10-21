
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.compare_models import (
    LABELS,
    ODDS_COLUMNS,
    WALK_FORWARD_FOLDS,
    aligned_probabilities,
    calculate_metrics,
    load_dataset,
    valid_odds_mask,
)
from backend.form_features import FEATURES


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Alpha = weight given to FootballIQ's form + odds model.
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def market_probabilities(df):
    """Normalized B365 probabilities in H, D, A order."""
    odds = df[ODDS_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(odds).all() or (odds <= 1.0).any():
        raise ValueError("Invalid bookmaker odds found.")

    implied = 1.0 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def main():
    dataset = load_dataset()
    all_results = []

    print("\nFootballIQ — probability blending experiment")
    print(f"Matches loaded: {len(dataset)}")
    print(f"Testing alpha values: {ALPHAS.tolist()}")

    for fold in WALK_FORWARD_FOLDS:
        train = dataset[
            dataset["season"].isin(fold["train"])
        ].copy()

        test_all = dataset[
            dataset["season"] == fold["test"]
        ].copy()

        # Same valid-odds test fixtures for all comparisons.
        test = test_all.loc[valid_odds_mask(test_all)].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"Empty train/test split for {fold['test']}."
            )

        # Train only on seasons before the test season.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        )

        model.fit(train[FEATURES], train["target"])

        # Our model's probabilities and the market's probabilities.
        p_model = aligned_probabilities(model, test[FEATURES])
        p_market = market_probabilities(test)

        y_true = test["target"].to_numpy()

        for alpha in ALPHAS:
            p_blend = (
                alpha * p_model
                + (1.0 - alpha) * p_market
            )

            predictions = np.array(LABELS)[
                p_blend.argmax(axis=1)
            ]

            metrics = calculate_metrics(
                y_true,
                predictions,
                p_blend,
            )

            all_results.append({
                "test_season": fold["test"],
                "alpha_model": float(alpha),
                "alpha_market": float(1.0 - alpha),
                "test_matches": len(test),
                **metrics,
            })

        print(
            f"Finished test season {fold['test']}: "
            f"{len(test)} matches"
        )

    results = pd.DataFrame(all_results)

    # Each alpha is evaluated over all held-out seasons.
    summary = (
        results.groupby(
            ["alpha_model", "alpha_market"],
            as_index=False,
        )
        .agg(
            total_test_matches=("test_matches", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
    )

    summary = summary.sort_values(
        ["mean_log_loss", "mean_macro_f1"],
        ascending=[True, False],
    ).reset_index(drop=True)

    details_path = REPORTS / "probability_blend_by_season.csv"
    summary_path = REPORTS / "probability_blend_summary.csv"

    results.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== BLENDING RESULTS ===")
    print(summary.round(4).to_string(index=False))

    print("\nBest blend by mean log loss:")
    print(summary.iloc[0].round(4).to_string())

    print(f"\nDetailed results: {details_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
