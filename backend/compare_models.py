
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.form_features import (
    FEATURES,
    build_fixture_features,
    load_historical_history,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "reports"
RESULTS.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "E0_2020_21.csv": "2020/21",
    "E0_2021_22.csv": "2021/22",
    "E0_2022_23.csv": "2022/23",
    "E0_2023_24.csv": "2023/24",
    "E0_2024_25.csv": "2024/25",
    "E0.csv": "2025/26",
}

LABELS = ["H", "D", "A"]

ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

MARKET_FEATURES = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]

CORE_FORM_FEATURES = [
    "home_points_avg",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_points_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
]

CLASS_NAMES = {
    "H": "Home win",
    "D": "Draw",
    "A": "Away win",
}

# Fixed across experiments so the feature comparison is fair.
CLASS_WEIGHTS = {"H": 1, "D": 1.5, "A": 1}

WALK_FORWARD_FOLDS = [
    {
        "train": ["2021/22", "2022/23"],
        "test": "2023/24",
    },
    {
        "train": ["2021/22", "2022/23", "2023/24"],
        "test": "2024/25",
    },
    {
        "train": [
            "2021/22",
            "2022/23",
            "2023/24",
            "2024/25",
        ],
        "test": "2025/26",
    },
]

FORM_MODELS = [
    "Prior baseline",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

COMBINED_MODELS = [
    "Form + odds (all features)",
    "Form + odds (core features)",
]

ODDS_MODEL = "Bookmaker odds (B365)"


# ============================================================
# DATA LOADING AND FEATURE ENGINEERING
# ============================================================

def load_dataset():
    """Build historical pre-match form features and attach odds."""

    history = load_historical_history()
    rows = []

    for filename, season in SEASONS.items():
        path = RAW / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        df = pd.read_csv(path)

        required = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"{filename} is missing required columns: {missing}"
            )

        # Parse dates once and discard invalid rows.
        df["parsed_date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce",
        )

        df = df.dropna(
            subset=["parsed_date", "HomeTeam", "AwayTeam", "FTR"]
        ).copy()

        df = df[df["FTR"].isin(LABELS)].copy()

        df = df.sort_values(
            ["parsed_date", "HomeTeam", "AwayTeam"]
        )

        print(
            f"Loaded {season}: {len(df)} matches "
            f"({df['parsed_date'].min().date()} to "
            f"{df['parsed_date'].max().date()})"
        )

        for _, match in df.iterrows():
            kickoff = match["parsed_date"].isoformat()

            features, _, _ = build_fixture_features(
                history=history,
                home_name=match["HomeTeam"],
                away_name=match["AwayTeam"],
                kickoff=kickoff,
            )

            row = {
                **features,
                "target": match["FTR"],
                "season": season,
                "date": match["parsed_date"],
                "home_team": match["HomeTeam"],
                "away_team": match["AwayTeam"],
            }

            # Missing bookmaker odds are retained as NaN.
            for col in ODDS_COLUMNS:
                if col in df.columns:
                    row[col] = pd.to_numeric(
                        match[col],
                        errors="coerce",
                    )
                else:
                    row[col] = np.nan

            rows.append(row)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        raise ValueError("No match data was loaded.")

    missing_features = set(FEATURES) - set(dataset.columns)

    if missing_features:
        raise ValueError(
            "Some configured form features were not generated: "
            f"{sorted(missing_features)}"
        )

    # Form features and targets are required for the form-only models.
    dataset = dataset.dropna(
        subset=list(FEATURES) + ["target"]
    ).copy()

    # Audit duplicate fixtures before evaluation.
    fixture_key = ["date", "home_team", "away_team"]

    duplicate_mask = dataset.duplicated(
        fixture_key,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    print(f"\nDuplicate fixture rows detected: {duplicate_count}")

    if duplicate_count:
        print(
            dataset.loc[
                duplicate_mask,
                fixture_key + ["season", "target"],
            ]
            .sort_values(fixture_key)
            .head(30)
            .to_string(index=False)
        )

        raise ValueError(
            "Duplicate fixtures detected. Resolve them before evaluation."
        )

    dataset = (
        dataset.sort_values(
            ["date", "home_team", "away_team"]
        )
        .reset_index(drop=True)
    )

    return dataset


def valid_odds_mask(df):
    """Return rows with all three valid decimal bookmaker odds."""

    mask = df[ODDS_COLUMNS].notna().all(axis=1)

    for col in ODDS_COLUMNS:
        mask &= df[col] > 1.0

    return mask


def add_market_features(df):
    """
    Convert bookmaker decimal odds into normalized implied probabilities.

    These are market-implied probabilities, not calibrated true
    probabilities. Normalization removes the bookmaker overround.
    """

    result = df.copy()

    valid = valid_odds_mask(result)

    result[MARKET_FEATURES] = np.nan

    if valid.any():
        inverse = 1.0 / result.loc[
            valid, ODDS_COLUMNS
        ].astype(float)

        total = inverse.sum(axis=1)

        probabilities = inverse.div(total, axis=0)

        result.loc[
            valid, MARKET_FEATURES
        ] = probabilities.to_numpy()

    return result


# ============================================================
# MODEL PROBABILITIES AND METRICS
# ============================================================

def aligned_probabilities(estimator, X):
    """Return predict_proba columns consistently ordered H, D, A."""

    raw = estimator.predict_proba(X)

    model_classes = list(estimator.classes_)

    aligned = np.zeros(
        (len(X), len(LABELS)),
        dtype=float,
    )

    for column, label in enumerate(model_classes):
        aligned[:, LABELS.index(label)] = raw[:, column]

    aligned = np.clip(aligned, 1e-15, 1.0)

    aligned /= aligned.sum(axis=1, keepdims=True)

    return aligned


def calculate_metrics(y_true, predictions, probabilities):
    """
    Calculate metrics with probability columns in H, D, A order.

    Scikit-learn's explicit log-loss label order is alphabetical:
    A, D, H. Reorder the probability columns accordingly.
    """

    log_order = [
        LABELS.index(label)
        for label in ["A", "D", "H"]
    ]

    ordered_probs = probabilities[:, log_order]

    y_array = np.asarray(y_true)

    y_one_hot = np.column_stack([
        (y_array == label).astype(float)
        for label in LABELS
    ])

    # Multiclass Brier score: mean of the summed squared errors.
    brier = np.mean(
        np.sum(
            (probabilities - y_one_hot) ** 2,
            axis=1,
        )
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_true,
            ordered_probs,
            labels=["A", "D", "H"],
        ),
        "brier_score": brier,
    }


def evaluate_predictions(
    model_name,
    season,
    y_true,
    predictions,
    probabilities,
    train_matches,
):
    """Print and collect metrics, class scores, and confusion matrix."""

    metrics = calculate_metrics(
        y_true,
        predictions,
        probabilities,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=LABELS,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=LABELS,
        target_names=[
            CLASS_NAMES[label]
            for label in LABELS
        ],
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{'=' * 70}")
    print(f"{model_name} — test season {season}")
    print("=" * 70)

    print(f"Training matches: {train_matches}")
    print(f"Test matches: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")

    print("\nConfusion matrix (actual rows; predicted columns):")

    print(
        pd.DataFrame(
            matrix,
            index=[
                f"Actual {CLASS_NAMES[label]}"
                for label in LABELS
            ],
            columns=[
                f"Pred {CLASS_NAMES[label]}"
                for label in LABELS
            ],
        )
    )

    print("\nClassification report:")
    print(
        pd.DataFrame(report)
        .T.round(3)
        .to_string()
    )

    result = {
        "model": model_name,
        "test_season": season,
        "train_matches": train_matches,
        "test_matches": len(y_true),
        **metrics,
    }

    class_rows = []

    for label in LABELS:
        values = report[CLASS_NAMES[label]]

        class_rows.append({
            "model": model_name,
            "test_season": season,
            "class": CLASS_NAMES[label],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })

    confusion_rows = []

    for i, actual in enumerate(LABELS):
        for j, predicted in enumerate(LABELS):
            confusion_rows.append({
                "model": model_name,
                "test_season": season,
                "actual": CLASS_NAMES[actual],
                "predicted": CLASS_NAMES[predicted],
                "count": int(matrix[i, j]),
            })

    return result, class_rows, confusion_rows


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():
    """Create form-only models for the current walk-forward fold."""

    return {
        "Prior baseline": DummyClassifier(
            strategy="prior",
        ),

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
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),

        "Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def create_combined_model():
    """Create the same model architecture for both feature experiments."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight=CLASS_WEIGHTS,
            random_state=42,
        ),
    )


def evaluate_model(
    name,
    estimator,
    train,
    test,
    features=FEATURES,
):
    """Fit a form-only model and evaluate it on the test season."""

    estimator.fit(
        train[features],
        train["target"],
    )

    probabilities = aligned_probabilities(
        estimator,
        test[features],
    )

    predictions = np.asarray(LABELS)[
        probabilities.argmax(axis=1)
    ]

    return evaluate_predictions(
        name,
        test["season"].iloc[0],
        test["target"].to_numpy(),
        predictions,
        probabilities,
        len(train),
    )


def evaluate_odds_baseline(test):
    """Use normalized bookmaker-implied probabilities directly."""

    probabilities = test[
        MARKET_FEATURES
    ].to_numpy(dtype=float)

    predictions = np.asarray(LABELS)[
        probabilities.argmax(axis=1)
    ]

    return evaluate_predictions(
        ODDS_MODEL,
        test["season"].iloc[0],
        test["target"].to_numpy(),
        predictions,
        probabilities,
        0,
    )


def evaluate_combined_experiment(
    model_name,
    features,
    train,
    test,
):
    """Fit and evaluate one form-plus-odds feature configuration."""

    model = create_combined_model()

    model.fit(
        train[features],
        train["target"],
    )

    probabilities = aligned_probabilities(
        model,
        test[features],
    )

    predictions = np.asarray(LABELS)[
        probabilities.argmax(axis=1)
    ]

    return evaluate_predictions(
        model_name,
        test["season"].iloc[0],
        test["target"].to_numpy(),
        predictions,
        probabilities,
        len(train),
    )


# ============================================================
# REPORT SAVING
# ============================================================

def save_reports(
    all_results,
    all_class_results,
    all_confusion_results,
):
    """Validate evaluation completeness and save all CSV reports."""

    results_df = pd.DataFrame(all_results)

    expected_models = (
        len(FORM_MODELS)
        + len(COMBINED_MODELS)
        + 1  # bookmaker odds baseline
    )

    expected_results = (
        len(WALK_FORWARD_FOLDS)
        * expected_models
    )

    if len(results_df) != expected_results:
        raise RuntimeError(
            f"Incomplete evaluation: expected {expected_results} "
            f"results, collected {len(results_df)}. "
            "Reports were not overwritten."
        )

    expected_seasons = {
        fold["test"]
        for fold in WALK_FORWARD_FOLDS
    }

    if set(results_df["test_season"].unique()) != expected_seasons:
        raise RuntimeError(
            "Unexpected test seasons. Reports were not overwritten."
        )

    if results_df.duplicated(
        ["model", "test_season"]
    ).any():
        raise RuntimeError(
            "Duplicate model/season results. Reports were not overwritten."
        )

    expected_model_names = (
        set(FORM_MODELS)
        | set(COMBINED_MODELS)
        | {ODDS_MODEL}
    )

    if set(results_df["model"].unique()) != expected_model_names:
        raise RuntimeError(
            "Unexpected model names in results. "
            "Reports were not overwritten."
        )

    summary = (
        results_df.groupby("model", as_index=False)
        .agg(
            seasons_tested=("test_season", "nunique"),
            total_test_matches=("test_matches", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .sort_values(
            ["mean_log_loss", "mean_brier_score"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        RESULTS / "walk_forward_results.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "walk_forward_summary.csv",
        index=False,
    )

    pd.DataFrame(all_class_results).to_csv(
        RESULTS / "walk_forward_class_metrics.csv",
        index=False,
    )

    pd.DataFrame(all_confusion_results).to_csv(
        RESULTS / "walk_forward_confusion_matrices.csv",
        index=False,
    )

    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("Lower log loss and Brier score are better.")
    print("Higher accuracy and macro F1 are better.")
    print("=" * 100)

    print(
        summary.round(4).to_string(index=False)
    )

    print(f"\nReports saved to: {RESULTS}")


# ============================================================
# MAIN WALK-FORWARD EVALUATION
# ============================================================

def main():
    # Build all historical form features, then add market probabilities.
    dataset = add_market_features(
        load_dataset()
    )

    # Validate the core feature names before starting any experiments.
    missing_core = set(CORE_FORM_FEATURES) - set(FEATURES)

    if missing_core:
        raise ValueError(
            "CORE_FORM_FEATURES contains names not present in "
            f"backend.form_features.FEATURES: {sorted(missing_core)}"
        )

    # Validate the generated market columns.
    missing_market = (
        set(MARKET_FEATURES) - set(dataset.columns)
    )

    if missing_market:
        raise ValueError(
            f"Market features were not generated: {sorted(missing_market)}"
        )

    # --------------------------------------------------------
    # DATA AUDIT
    # --------------------------------------------------------

    print("\nODDS PROBABILITY AUDIT")

    print(
        dataset[
            [
                "B365H",
                "B365D",
                "B365A",
                "market_home_prob",
                "market_draw_prob",
                "market_away_prob",
                "target",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nAverage market probabilities:")

    print(
        dataset[MARKET_FEATURES]
        .mean()
        .round(3)
        .to_string()
    )

    valid_market = dataset[MARKET_FEATURES].dropna()

    if not valid_market.empty:
        print("\nPredicted bookmaker outcomes:")

        print(
            valid_market.idxmax(axis=1)
            .value_counts()
            .to_string()
        )

    print("\nActual outcomes:")

    print(
        dataset["target"]
        .value_counts()
        .to_string()
    )

    print("\nFootballIQ — Premier League walk-forward evaluation")
    print(f"Total matches loaded: {len(dataset)}")
    print(f"Form features: {len(FEATURES)}")
    print(f"Core form features: {len(CORE_FORM_FEATURES)}")
    print(f"Combined features, full: {len(FEATURES) + len(MARKET_FEATURES)}")
    print(f"Combined features, core: {len(CORE_FORM_FEATURES) + len(MARKET_FEATURES)}")

    print("\nMatches per season:")

    print(
        dataset.groupby("season")
        .size()
        .to_string()
    )

    # Both feature experiments use the same training/test folds.
    experiments = {
        "Form + odds (all features)": (
            list(FEATURES) + MARKET_FEATURES
        ),
        "Form + odds (core features)": (
            CORE_FORM_FEATURES + MARKET_FEATURES
        ),
    }

    all_results = []
    all_class_results = []
    all_confusion_results = []

    # --------------------------------------------------------
    # WALK-FORWARD FOLDS
    # --------------------------------------------------------

    for fold in WALK_FORWARD_FOLDS:
        train_all = dataset[
            dataset["season"].isin(fold["train"])
        ].copy()

        test_all = dataset[
            dataset["season"] == fold["test"]
        ].copy()

        if train_all.empty or test_all.empty:
            raise ValueError(
                f"Empty train/test split for {fold['test']}."
            )

        # All models are evaluated on the same test fixtures,
        # restricted to fixtures with valid bookmaker odds.
        test = test_all.loc[
            valid_odds_mask(test_all)
        ].copy()

        if test.empty:
            raise ValueError(
                f"No valid odds in test season {fold['test']}."
            )

        # Form-only models can use every valid historical form row.
        # Form + odds models require valid odds in training too.
        train_with_odds = train_all.loc[
            valid_odds_mask(train_all)
        ].copy()

        if train_with_odds.empty:
            raise ValueError(
                f"No valid training odds before {fold['test']}."
            )

        print("\n\n" + "#" * 75)
        print(f"TRAIN SEASONS: {fold['train']}")
        print(f"TEST SEASON: {fold['test']}")
        print("#" * 75)

        print(f"Training matches (form models): {len(train_all)}")
        print(f"Training matches with valid odds: {len(train_with_odds)}")
        print(f"Common test matches with valid odds: {len(test)}")
        print(
            "Test matches excluded for invalid odds: "
            f"{len(test_all) - len(test)}"
        )

        # ----------------------------------------------------
        # 1. FORM-ONLY MODELS
        # ----------------------------------------------------

        for name, estimator in create_models().items():
            result, class_rows, confusion_rows = evaluate_model(
                name,
                estimator,
                train_all,
                test,
                FEATURES,
            )

            all_results.append(result)
            all_class_results.extend(class_rows)
            all_confusion_results.extend(confusion_rows)

        # ----------------------------------------------------
        # 2. FORM + ODDS EXPERIMENTS
        # ----------------------------------------------------

        for model_name, selected_features in experiments.items():
            result, class_rows, confusion_rows = (
                evaluate_combined_experiment(
                    model_name,
                    selected_features,
                    train_with_odds,
                    test,
                )
            )

            all_results.append(result)
            all_class_results.extend(class_rows)
            all_confusion_results.extend(confusion_rows)

        # ----------------------------------------------------
        # 3. BOOKMAKER ODDS BASELINE
        # ----------------------------------------------------

        result, class_rows, confusion_rows = (
            evaluate_odds_baseline(test)
        )

        all_results.append(result)
        all_class_results.extend(class_rows)
        all_confusion_results.extend(confusion_rows)

        expected_so_far = (
            WALK_FORWARD_FOLDS.index(fold) + 1
        ) * (
            len(FORM_MODELS)
            + len(COMBINED_MODELS)
            + 1
        )

        print(
            f"\nResults collected: {len(all_results)} / "
            f"{len(WALK_FORWARD_FOLDS) * (len(FORM_MODELS) + len(COMBINED_MODELS) + 1)}"
        )

        if len(all_results) != expected_so_far:
            raise RuntimeError(
                f"Unexpected result count after {fold['test']}."
            )

    # Only save reports after every fold and model has completed.
    save_reports(
        all_results,
        all_class_results,
        all_confusion_results,
    )


if __name__ == "__main__":
    main()
