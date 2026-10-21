
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from backend.compare_models import load_dataset, valid_odds_mask
from backend.form_features import FEATURES as FORM_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = MODEL_DIR / "footballiq_form_odds_model.joblib"

ODDS = ["B365H", "B365D", "B365A"]
MARKET_FEATURES = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]


def main():
    data = load_dataset().copy()

    # Train only on rows with valid odds because this model requires them.
    data = data.loc[valid_odds_mask(data)].copy()

    inverse = 1.0 / data[ODDS].astype(float)
    total = inverse.sum(axis=1)

    data["market_home_prob"] = inverse["B365H"] / total
    data["market_draw_prob"] = inverse["B365D"] / total
    data["market_away_prob"] = inverse["B365A"] / total

    features = list(FORM_FEATURES) + MARKET_FEATURES

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    model.fit(data[features], data["target"])

    artifact = {
        "model": model,
        "features": features,
        "form_features": list(FORM_FEATURES),
        "market_features": MARKET_FEATURES,
        "target_classes": list(model[-1].classes_),
        "training_rows": len(data),
        "training_seasons": sorted(data["season"].unique().tolist()),
        "requires_odds": True,
    }

    joblib.dump(artifact, OUTPUT_PATH)

    print(f"Saved model: {OUTPUT_PATH}")
    print(f"Training rows: {len(data)}")
    print(f"Features: {len(features)}")
    print(f"Seasons: {artifact['training_seasons']}")


if __name__ == "__main__":
    main()
