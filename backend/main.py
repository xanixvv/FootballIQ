
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict, deque
from backend.ucl_team_names import normalize_ucl_team_name

import math
import sqlite3
import time

from backend.form_features import (
    FEATURES as FORM_FEATURES,
    build_fixture_features,
    build_team_form,
    load_historical_history,
)
from backend.odds_provider import fetch_odds


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"

MODEL_PATH = PROJECT_ROOT / "models" / "outcome_model.joblib"
FORM_ODDS_MODEL_PATH = (
    PROJECT_ROOT / "models" / "footballiq_form_odds_model.joblib"
)

# --------------------------------------------------
# CHAMPIONS LEAGUE MODEL
# --------------------------------------------------

UCL_MODEL_PATH = PROJECT_ROOT / "models" / "footballiq_ucl_model.joblib"
UCL_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "ucl_matches.csv"

ucl_artifact = None
ucl_model = None
UCL_FEATURES = []
ucl_history = pd.DataFrame()
ucl_model_load_error = None

try:
    ucl_artifact = joblib.load(UCL_MODEL_PATH)
    ucl_model = ucl_artifact["model"]
    UCL_FEATURES = list(ucl_artifact["features"])

    required_ucl_features = {
        "home_points_avg",
        "home_goals_for_avg",
        "home_goals_against_avg",
        "home_win_rate",
        "away_points_avg",
        "away_goals_for_avg",
        "away_goals_against_avg",
        "away_win_rate",
        "home_matches_played",
        "away_matches_played",
        "home_advantage",
    }

    if set(UCL_FEATURES) != required_ucl_features:
        raise ValueError(
            f"Unexpected UCL model features: {UCL_FEATURES}"
        )

    ucl_history = pd.read_csv(UCL_HISTORY_PATH)

    required_columns = {
        "date", "home_team", "away_team",
        "home_goals", "away_goals", "result",
    }
    missing_columns = required_columns - set(ucl_history.columns)

    if missing_columns:
        raise ValueError(
            f"UCL history is missing columns: {sorted(missing_columns)}"
        )

    ucl_history["date"] = pd.to_datetime(
        ucl_history["date"], errors="coerce", utc=True
    )
    ucl_history["home_goals"] = pd.to_numeric(
        ucl_history["home_goals"], errors="coerce"
    )
    ucl_history["away_goals"] = pd.to_numeric(
        ucl_history["away_goals"], errors="coerce"
    )

    ucl_history = ucl_history.dropna(
        subset=[
            "date", "home_team", "away_team",
            "home_goals", "away_goals", "result",
        ]
    ).copy()

    ucl_history = ucl_history[
        ucl_history["result"].isin(["H", "D", "A"])
    ].copy()

    ucl_history["home_key"] = ucl_history["home_team"].map(
        normalize_ucl_team_name
    )
    ucl_history["away_key"] = ucl_history["away_team"].map(
        normalize_ucl_team_name
    )

    print(
        f"UCL model loaded: {UCL_MODEL_PATH.name}; "
        f"{len(ucl_history)} historical matches."
    )

except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
    ucl_model_load_error = str(exc)
    ucl_model = None
    print(f"UCL model unavailable: {ucl_model_load_error}")
# Existing form-only model
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

# Combined form + bookmaker odds model
form_odds_artifact = joblib.load(FORM_ODDS_MODEL_PATH)
form_odds_model = form_odds_artifact["model"]
FORM_ODDS_FEATURES = form_odds_artifact["features"]
FORM_FEATURE_NAMES = form_odds_artifact["form_features"]
MARKET_FEATURE_NAMES = form_odds_artifact["market_features"]


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="FootballIQ API",
    description="Football match analytics and ML predictions",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://football-8clt60y8c-xanixvvs-projects.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FootballIQ: live match scores from API-Football

# Experimental in-play baseline and snapshot history.
# This is NOT yet a trained or calibrated ML model.

LIVE_DB = Path(__file__).resolve().parent / "live_predictions.db"
LIVE_CACHE_SECONDS = 15 * 60
_live_cache = {"time": 0.0, "data": None}


def init_live_db():
    with sqlite3.connect(LIVE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER NOT NULL,
                minute INTEGER,
                home_goals INTEGER,
                away_goals INTEGER,
                home_win REAL NOT NULL,
                draw REAL NOT NULL,
                away_win REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE(fixture_id, minute, home_goals, away_goals)
            )
        """)


def estimate_in_play_baseline(home_goals, away_goals, minute):
    """
    Simple experimental score-and-time heuristic.
    These probabilities are illustrative estimates, not validated forecasts.
    """
    home_goals = int(home_goals or 0)
    away_goals = int(away_goals or 0)
    minute = max(0, min(int(minute or 0), 120))

    progress = min(minute / 90.0, 1.0)
    remaining = max(0.05, 1.0 - progress)

    # Score difference matters more as the match approaches its end.
    score_effect = (home_goals - away_goals) * (0.65 + 1.8 * progress)
    home_advantage = 0.22 * remaining

    home_strength = math.exp(max(-5, min(5, score_effect + home_advantage)))
    away_strength = math.exp(max(-5, min(5, -score_effect)))

    # Draw weight decreases as time runs out, and rises when level.
    draw_strength = 0.9 * remaining + (
        0.75 if home_goals == away_goals else 0.12 * remaining
    )

    total = home_strength + away_strength + draw_strength

    return {
        "home": round(home_strength / total, 4),
        "draw": round(draw_strength / total, 4),
        "away": round(away_strength / total, 4),
        "model": "experimental_score_time_baseline",
        "validated": False,
    }


def save_live_snapshot(match, prediction):
    minute = match.get("minute")
    if minute is None:
        return

    home_goals = int(match.get("home_goals") or 0)
    away_goals = int(match.get("away_goals") or 0)

    with sqlite3.connect(LIVE_DB) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO live_snapshots (
                fixture_id, minute, home_goals, away_goals,
                home_win, draw, away_win, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match["fixture_id"],
            int(minute),
            home_goals,
            away_goals,
            prediction["home"],
            prediction["draw"],
            prediction["away"],
            datetime.now(timezone.utc).isoformat(),
        ))


def get_live_snapshot_history(fixture_id, limit=5):
    with sqlite3.connect(LIVE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT minute, home_goals, away_goals,
                   home_win, draw, away_win, recorded_at
            FROM live_snapshots
            WHERE fixture_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (fixture_id, limit)).fetchall()

    history = [dict(row) for row in rows]
    history.reverse()
    return history


init_live_db()


@app.get("/live/matches")
def get_live_matches():
    api_key = os.getenv("API_FOOTBALL_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API_FOOTBALL_KEY is missing from backend/.env"
        )

    try:
        response = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"live": "all"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not reach the football data provider."
        )

    if data.get("errors"):
        raise HTTPException(
            status_code=502,
            detail={"provider_errors": data["errors"]}
        )

    matches = []

    for item in data.get("response", []):
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        league = item.get("league", {})
        status = fixture.get("status", {})

        matches.append({
            "fixture_id": fixture.get("id"),
            "league": league.get("name"),
            "country": league.get("country"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "home_logo": teams.get("home", {}).get("logo"),
            "away_logo": teams.get("away", {}).get("logo"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "minute": status.get("elapsed"),
            "status": status.get("long"),
            "status_short": status.get("short"),
            "kickoff": fixture.get("date"),
        })

    return {
        "count": len(matches),
        "matches": matches,
        "source": "API-Football",
        "is_live_data": True,
    }


@app.get("/live/predictions")
def get_live_predictions():
    # Share one provider response across frontend refreshes.
    now = time.time()
    cached = _live_cache["data"]

    if cached is not None and now - _live_cache["time"] < LIVE_CACHE_SECONDS:
        return cached

    live_data = get_live_matches()

    for match in live_data.get("matches", []):
        prediction = estimate_in_play_baseline(
            match.get("home_goals"),
            match.get("away_goals"),
            match.get("minute"),
        )

        match["in_play_prediction"] = prediction
        save_live_snapshot(match, prediction)
        match["prediction_history"] = get_live_snapshot_history(
            match["fixture_id"], limit=5
        )

    result = {
        **live_data,
        "prediction_type": "experimental_baseline",
        "prediction_warning": (
            "Experimental score-and-time heuristic; not a trained or calibrated "
            "in-play ML model. Do not treat probabilities as reliable forecasts."
        ),
        "snapshot_limit": 5,
    }

    _live_cache["time"] = now
    _live_cache["data"] = result
    return result


# --------------------------------------------------
# PLAYER MARKET VALUE MODEL
# --------------------------------------------------

PLAYER_MODEL_PATH = PROJECT_ROOT / "models" / "small_player_value_30trees_depth12.joblib"
PLAYER_METADATA_PATH = PROJECT_ROOT / "models" / "player_market_value_metadata.json"
PLAYER_PROFILES_PATH = PROJECT_ROOT / "data" / "raw" / "players.csv"
PLAYER_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_market_value_with_history.csv"
)
# Download the player market-value model from private Hugging Face
# only when the model file is missing on this machine.
if not PLAYER_MODEL_PATH.exists():
    try:
        from huggingface_hub import hf_hub_download

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is missing from the environment."
            )

        PLAYER_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Downloading player market-value model from Hugging Face...")

        hf_hub_download(
            repo_id="xanixvv/footballiq-player-model",
            filename="small_player_value_30trees_depth12.joblib",
            token=hf_token,
            local_dir=str(PLAYER_MODEL_PATH.parent),
        )

        print("Player market-value model downloaded successfully.")

    except Exception as exc:
        print(
            f"Player market-value model download failed: {exc}"
        )
# Keep the match-prediction API usable even if the optional player model has
# not been trained or its data files have not been created yet. The player
# endpoints return a clear 503 in that case.
player_value_model = None
player_value_metadata = {}
player_features = []
player_profiles = pd.DataFrame()
player_history = pd.DataFrame()
player_value_load_error = None

print("DEBUG: Starting player model initialization", flush=True)

try:
    print("DEBUG: About to load player model:", PLAYER_MODEL_PATH, flush=True)
    player_value_model = joblib.load(PLAYER_MODEL_PATH)
    with PLAYER_METADATA_PATH.open("r", encoding="utf-8") as file:
        player_value_metadata = json.load(file)
    player_features = player_value_metadata.get("features", [])

    player_profiles = pd.read_csv(PLAYER_PROFILES_PATH)
    player_history = pd.read_csv(PLAYER_HISTORY_PATH, parse_dates=["date"])

    if "player_id" not in player_profiles.columns:
        raise ValueError("players.csv must contain a player_id column.")
    if not {"player_id", "date"}.issubset(player_history.columns):
        raise ValueError(
            "Player history CSV must contain player_id and date columns."
        )

    player_profiles["player_id"] = pd.to_numeric(
        player_profiles["player_id"], errors="coerce"
    )
    player_history["player_id"] = pd.to_numeric(
        player_history["player_id"], errors="coerce"
    )

    player_profiles = player_profiles.dropna(subset=["player_id"]).copy()
    player_history = player_history.dropna(subset=["player_id", "date"]).copy()

    player_profiles["player_id"] = player_profiles["player_id"].astype("int64")
    player_history["player_id"] = player_history["player_id"].astype("int64")
    player_history = (
        player_history.sort_values("date")
        .drop_duplicates(subset=["player_id"], keep="last")
        .copy()
    )
except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
    player_value_load_error = str(exc)
    print(f"Player market-value model unavailable: {player_value_load_error}")


    
# --------------------------------------------------
# PLAYER VALUE FORECAST + DIRECTION MODELS
# --------------------------------------------------

PLAYER_FORECAST_PATH = (
    PROJECT_ROOT / "models" / "player_value_forecast_v2.joblib"
)
PLAYER_DIRECTION_PATH = (
    PROJECT_ROOT / "models" / "player_value_direction_model.joblib"
)

player_forecast_artifact = None
player_direction_artifact = None
player_forecast_load_error = None
player_direction_load_error = None

try:
    player_forecast_artifact = joblib.load(PLAYER_FORECAST_PATH)
except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
    player_forecast_load_error = str(exc)
    print(f"Player value forecast unavailable: {exc}")

try:
    player_direction_artifact = joblib.load(PLAYER_DIRECTION_PATH)
except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
    player_direction_load_error = str(exc)
    print(f"Player value direction model unavailable: {exc}")


def require_player_prediction_data():
    if player_profiles.empty or player_history.empty:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Player prediction data is unavailable.",
                "reason": player_value_load_error
                or "Player profiles or historical features are missing.",
            },
        )


def get_player_prediction_row(player_id: int, numeric_features, categorical_features):
    require_player_prediction_data()

    profiles = player_profiles[player_profiles["player_id"] == player_id]
    history = player_history[player_history["player_id"] == player_id]

    if profiles.empty:
        raise HTTPException(status_code=404, detail="Player not found.")

    if history.empty:
        raise HTTPException(
            status_code=404,
            detail="No historical feature row exists for this player.",
        )

    latest = history.sort_values("date").iloc[-1]
    feature_names = list(numeric_features) + list(categorical_features)
    missing = [name for name in feature_names if name not in latest.index]

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Required prediction features are missing.",
                "missing_features": missing,
            },
        )

    input_data = pd.DataFrame(
        [[latest[name] for name in feature_names]],
        columns=feature_names,
    )

    return profiles.iloc[0], latest, input_data



def require_player_model():
    """Raise a clear API error when player-model artifacts are unavailable."""
    if (
        player_value_model is None
        or player_profiles.empty
        or player_history.empty
        or not player_features
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Player market-value model is not ready.",
                "reason": player_value_load_error
                or "One or more player model/data files are missing or empty.",
            },
        )


def _clean_value(value):
    """Convert pandas/numpy missing values into JSON-friendly None values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def player_profile_summary(player):
    return {
        "player_id": int(player["player_id"]),
        "name": _clean_value(player.get("name")) or "Unknown player",
        "position": _clean_value(player.get("position")),
        "sub_position": _clean_value(player.get("sub_position")),
        "foot": _clean_value(player.get("foot")),
        "height": _clean_value(
            player.get("height_in_cm", player.get("height"))
        ),
        "date_of_birth": _clean_value(player.get("date_of_birth")),
        "nationality": _clean_value(
            player.get("country_of_citizenship")
        ),
        "current_club": _clean_value(
            player.get("current_club_name", player.get("current_club"))
        ),
        "current_club_id": _clean_value(player.get("current_club_id")),
        "image_url": _clean_value(player.get("image_url")),
        "current_market_value": _clean_value(
            player.get(
                "market_value_in_eur",
                player.get("current_market_value")
            )
        ),
        "highest_market_value": _clean_value(
            player.get("highest_market_value_in_eur")
        ),
    }

def normalize_search_text(value):
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(
        char for char in value
        if not unicodedata.combining(char)
    ).casefold()



@app.get("/players/search")
def search_players(q: str, limit: int = 10):
    """Accent-insensitive player search."""
    query = q.strip()

    if len(query) < 2:
        raise HTTPException(
            status_code=422,
            detail="Search query must contain at least 2 characters.",
        )

    require_player_model()

    normalized_query = normalize_search_text(query)
    normalized_names = player_profiles["name"].fillna("").map(
        normalize_search_text
    )

    matches = player_profiles[
        normalized_names.str.contains(
            re.escape(normalized_query),
            na=False,
            regex=True,
        )
    ]

    limit = max(1, min(limit, 50))
    matches = matches.head(limit)

    return {
        "query": q,
        "count": int(len(matches)),
        "players": [
            player_profile_summary(row)
            for _, row in matches.iterrows()
        ],
    }

@app.get("/players/{player_id}/market-value")
def predict_player_market_value(player_id: int):
    """Estimate a player's market value using the latest saved feature row."""
    require_player_model()

    profiles = player_profiles[player_profiles["player_id"] == player_id]
    history = player_history[player_history["player_id"] == player_id]

    if profiles.empty:
        raise HTTPException(status_code=404, detail="Player not found.")
    if history.empty:
        raise HTTPException(
            status_code=404,
            detail="No historical feature row exists for this player.",
        )

    profile = profiles.iloc[0]
    latest = history.sort_values("date").iloc[-1]
    missing_features = [name for name in player_features if name not in latest.index]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Player model features do not match the history data.",
                "missing_features": missing_features,
            },
        )

    try:
        input_data = pd.DataFrame(
            [[latest[name] for name in player_features]],
            columns=player_features,
        )
        raw_prediction = float(
            np.asarray(player_value_model.predict(input_data)).reshape(-1)[0]
        )

        predicted_value_eur = max(0.0, float(np.expm1(raw_prediction)))

        # The model was trained on log1p(market_value_in_eur).
        # Convert its prediction back into euros.
        estimate = float(np.expm1(raw_prediction))

        # Market value cannot be negative.
        estimate = max(0.0, estimate)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate player market-value estimate: {exc}",
        ) from exc

    return {
        "player": player_profile_summary(profile),
        "estimated_market_value": estimate,
        "currency": player_value_metadata.get("currency", "EUR"),
        "feature_data_date": _clean_value(latest.get("date")),
        "model_type": player_value_metadata.get(
            "model_type", type(player_value_model).__name__
        ),
        "metrics": player_value_metadata.get("metrics"),
        "status": "experimental",
        "disclaimer": (
            "This is an experimental model estimate, not an official valuation "
            "or a guaranteed transfer fee."
        ),
    }

@app.get("/players/{player_id}/market-value/latest-reported")
def latest_reported_player_market_value(player_id: int):
    """Return the latest recorded market valuation in our stored dataset."""
    require_player_prediction_data()

    profiles = player_profiles[
        player_profiles["player_id"] == player_id
    ]
    history = player_history[
        player_history["player_id"] == player_id
    ]

    if profiles.empty:
        raise HTTPException(status_code=404, detail="Player not found.")

    if history.empty:
        raise HTTPException(
            status_code=404,
            detail="No valuation history exists for this player.",
        )

    latest = history.sort_values("date").iloc[-1]

    # This is the recorded value, not an ML-generated estimate.
    reported_value = latest.get("market_value_in_eur")

    if reported_value is None or pd.isna(reported_value):
        raise HTTPException(
            status_code=404,
            detail=(
                "No recorded market value is available for this player "
                "in the stored dataset."
            ),
        )

    valuation_date = pd.Timestamp(latest["date"]).date()
    data_age_days = (
        datetime.now(timezone.utc).date() - valuation_date
    ).days

    return {
        "player": player_profile_summary(profiles.iloc[0]),
        "latest_reported_market_value_eur": float(reported_value),
        "currency": "EUR",
        "valuation_date": valuation_date.isoformat(),
        "data_age_days": data_age_days,
        "data_source": "FootballIQ stored player valuation dataset",
        "is_live": False,
        "status": "historical_record",
        "note": (
            "This is the latest valuation in the currently stored dataset. "
            "It may be outdated and is not necessarily the player's "
            "current market value."
        ),
    }


@app.get("/players/{player_id}/market-value/forecast")
def forecast_player_market_value(player_id: int):
    """Forecast the next recorded valuation 90–120 days after the feature date."""
    if player_forecast_artifact is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Player valuation forecast model is unavailable.",
                "reason": player_forecast_load_error,
            },
        )

    artifact = player_forecast_artifact

    profile, latest, input_data = get_player_prediction_row(
        player_id,
        artifact["numeric_features"],
        artifact["categorical_features"],
    )

    try:
        log_prediction = float(
            np.asarray(artifact["model"].predict(input_data)).reshape(-1)[0]
        )
        estimate = max(0.0, float(np.expm1(log_prediction)))
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not forecast player valuation: {exc}",
        ) from exc

    previous_value = _clean_value(latest.get("previous_value_eur"))

    return {
        "player": player_profile_summary(profile),
        "forecast_market_value_eur": round(estimate, 2),
        "previous_recorded_value_eur": previous_value,
        "currency": "EUR",
        "feature_data_date": _clean_value(latest.get("date")),
        "forecast_horizon_days": {
            "minimum": artifact["min_days"],
            "maximum": artifact["max_days"],
        },
        "latest_labeled_target_date": artifact["training_cutoff"],
        "model_type": type(
            artifact["model"].named_steps["regressor"]
        ).__name__,
        "status": "experimental",
        "disclaimer": (
            "Forecast of the next recorded valuation 90–120 days after "
            "the feature date. It is not a live valuation, official price, "
            "or guaranteed transfer fee."
        ),
    }


@app.get("/players/{player_id}/market-value/direction")
def predict_player_value_direction(player_id: int):
    """Predict whether the next recorded valuation rises, falls, or stays similar."""
    if player_direction_artifact is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Player value direction model is unavailable.",
                "reason": player_direction_load_error,
            },
        )

    artifact = player_direction_artifact

    profile, latest, input_data = get_player_prediction_row(
        player_id,
        artifact["numeric_features"],
        artifact["categorical_features"],
    )

    try:
        direction_model = artifact["model"]
        direction = str(direction_model.predict(input_data)[0])
        probabilities = direction_model.predict_proba(input_data)[0]
        classifier = (
            direction_model[-1]
            if hasattr(direction_model, "__getitem__")
            else direction_model
        )

        probability_map = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classifier.classes_, probabilities)
        }
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not predict valuation direction: {exc}",
        ) from exc

    return {
        "player": player_profile_summary(profile),
        "predicted_direction": direction,
        "class_probabilities": probability_map,
        "previous_recorded_value_eur": _clean_value(
            latest.get("previous_value_eur")
        ),
        "feature_data_date": _clean_value(latest.get("date")),
        "forecast_horizon_days": {
            "minimum": artifact["min_days"],
            "maximum": artifact["max_days"],
        },
        "latest_labeled_target_date": artifact["training_cutoff"],
        "model_type": artifact["model_name"],
        "status": "experimental",
        "disclaimer": (
            "Experimental prediction based on historical data. "
            "Class probabilities are not guaranteed to be calibrated."
        ),
    }



class PredictionRequest(BaseModel):
    features: dict[str, float]


class FormOddsPredictionRequest(BaseModel):
    features: dict[str, float]
    home_odds: float
    draw_odds: float
    away_odds: float


SUPPORTED_COMPETITIONS = {
    "PL": "Premier League",
    "CL": "UEFA Champions League",
}


# --------------------------------------------------
# TEAM-NAME MATCHING
# --------------------------------------------------

def normalize_team_name(name: str) -> str:
    """Normalize common naming differences between data providers."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )
    name = name.casefold().strip()
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    aliases = {
        "arsenal fc": "arsenal",
        "manchester city fc": "manchester city",
        "man city": "manchester city",
        "manchester united fc": "manchester united",
        "man utd": "manchester united",
        "tottenham hotspur fc": "tottenham hotspur",
        "spurs": "tottenham hotspur",
        "brighton and hove albion": "brighton",
        "brighton and hove albion fc": "brighton",
        "wolverhampton wanderers": "wolves",
        "wolverhampton wanderers fc": "wolves",
        "west ham united fc": "west ham",
        "newcastle united fc": "newcastle united",
        "nottingham forest fc": "nottingham forest",
        "afc bournemouth": "bournemouth",
        "bournemouth fc": "bournemouth",
        "leeds united fc": "leeds united",
        "sunderland afc": "sunderland",
        "fulham fc": "fulham",
        "chelsea fc": "chelsea",
        "liverpool fc": "liverpool",
        "everton fc": "everton",
        "brentford fc": "brentford",
        "crystal palace fc": "crystal palace",
        "aston villa fc": "aston villa",
        "ipswich town fc": "ipswich town",
        "manchester city": "manchester city",
        "manchester united": "manchester united",
        "brighton hove albion": "brighton",
    }

    return aliases.get(name, name)


def kickoff_times_match(fixture_kickoff: str, odds_kickoff: str) -> bool:
    """Avoid attaching odds to a different fixture or kickoff."""
    try:
        fixture_dt = datetime.fromisoformat(
            fixture_kickoff.replace("Z", "+00:00")
        )
        odds_dt = datetime.fromisoformat(
            odds_kickoff.replace("Z", "+00:00")
        )

        if fixture_dt.tzinfo is None:
            fixture_dt = fixture_dt.replace(tzinfo=timezone.utc)

        if odds_dt.tzinfo is None:
            odds_dt = odds_dt.replace(tzinfo=timezone.utc)

        return abs((fixture_dt - odds_dt).total_seconds()) <= 18 * 60 * 60

    except (TypeError, ValueError, AttributeError):
        return False


def find_matching_odds(match: dict, odds_events: list[dict]):
    """Return odds only when both teams and kickoff match."""
    fixture_home = normalize_team_name(match["home_team"])
    fixture_away = normalize_team_name(match["away_team"])

    for event in odds_events:
        odds_home = normalize_team_name(event.get("home_team", ""))
        odds_away = normalize_team_name(event.get("away_team", ""))

        if odds_home != fixture_home or odds_away != fixture_away:
            continue

        if not kickoff_times_match(
            match.get("kickoff", ""),
            event.get("kickoff", ""),
        ):
            continue

        try:
            odds = [
                float(event["home_decimal_odds"]),
                float(event["draw_decimal_odds"]),
                float(event["away_decimal_odds"]),
            ]
        except (KeyError, TypeError, ValueError):
            continue

        if any(not 1.0 < odd < 1000 for odd in odds):
            continue

        return event

    return None


# --------------------------------------------------
# PROBABILITY HELPERS
# --------------------------------------------------

def decimal_odds_to_probabilities(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> dict[str, float]:
    """Convert decimal odds to normalized implied probabilities."""
    odds = [home_odds, draw_odds, away_odds]

    if any(not 1.0 < odd < 1000 for odd in odds):
        raise ValueError("Decimal odds must be greater than 1 and less than 1000.")

    inverse = [1.0 / odd for odd in odds]
    total = sum(inverse)

    return {
        "market_home_prob": inverse[0] / total,
        "market_draw_prob": inverse[1] / total,
        "market_away_prob": inverse[2] / total,
    }


def predict_with_form_and_odds(
    features: dict[str, float],
    odds: dict,
) -> dict:
    """Generate a prediction using form features and real bookmaker odds."""
    missing = [
        name for name in FORM_FEATURE_NAMES
        if name not in features or features[name] is None
    ]

    if missing:
        raise ValueError(
            f"Missing form features for combined model: {missing}"
        )

    market = decimal_odds_to_probabilities(
        float(odds["home_decimal_odds"]),
        float(odds["draw_decimal_odds"]),
        float(odds["away_decimal_odds"]),
    )

    combined_features = {
        **{name: features[name] for name in FORM_FEATURE_NAMES},
        **market,
    }

    missing_combined = [
        name for name in FORM_ODDS_FEATURES
        if name not in combined_features
    ]

    if missing_combined:
        raise ValueError(
            f"Combined model is missing features: {missing_combined}"
        )

    input_data = pd.DataFrame(
        [[combined_features[name] for name in FORM_ODDS_FEATURES]],
        columns=FORM_ODDS_FEATURES,
    )

    probabilities = form_odds_model.predict_proba(input_data)[0]

    # The artifact may contain a Pipeline, so read classes from its classifier.
    classifier = (
        form_odds_model[-1]
        if hasattr(form_odds_model, "__getitem__")
        else form_odds_model
    )

    classes = classifier.classes_

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "predicted_outcome": str(form_odds_model.predict(input_data)[0]),
        "probabilities": {
            "home_win": round(probability_map.get("H", 0.0), 4),
            "draw": round(probability_map.get("D", 0.0), 4),
            "away_win": round(probability_map.get("A", 0.0), 4),
        },
        "market_probabilities": {
            "home_win": round(market["market_home_prob"], 4),
            "draw": round(market["market_draw_prob"], 4),
            "away_win": round(market["market_away_prob"], 4),
        },
        "model_type": "LogisticRegression (form + bookmaker odds)",
        "model_version": "footballiq_form_odds_model.joblib",
        "status": "experimental",
    }


def predict_form_only(features: dict[str, float]) -> dict:
    """Fallback prediction when suitable odds are unavailable."""
    missing = [
        name for name in FEATURES
        if name not in features or features[name] is None
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Feature builder and saved model do not match.",
                "missing_features": missing,
                "model_features": FEATURES,
                "available_features": list(features.keys()),
            },
        )

    input_data = pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    probabilities = model.predict_proba(input_data)[0]
    probability_map = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "predicted_outcome": str(model.predict(input_data)[0]),
        "probabilities": {
            "home_win": round(probability_map.get("H", 0.0), 4),
            "draw": round(probability_map.get("D", 0.0), 4),
            "away_win": round(probability_map.get("A", 0.0), 4),
        },
        "model_type": "Form-only model",
        "model_version": "outcome_model.joblib",
        "status": "experimental",
    }


# --------------------------------------------------
# HEALTH AND MODEL INFO
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "FootballIQ",
        "model_loaded": True,
        "form_odds_model_loaded": True,
        "player_market_value_model_loaded": player_value_model is not None,
        "ucl_model_loaded": ucl_model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features_required": FEATURES,
        "number_of_features": len(FEATURES),
        "outcome_classes": [str(c) for c in model.classes_],
        "form_odds_model_type": type(form_odds_model).__name__,
        "form_odds_features": FORM_ODDS_FEATURES,
        "form_odds_feature_count": len(FORM_ODDS_FEATURES),
        "player_market_value_model_loaded": player_value_model is not None,
        "player_market_value_features": player_features,
        "ucl_model_loaded": ucl_model is not None,
        "ucl_model_type": (
            type(ucl_model[-1]).__name__
            if ucl_model is not None and hasattr(ucl_model, "__getitem__")
            else type(ucl_model).__name__ if ucl_model is not None else None
        ),
        "ucl_model_features": UCL_FEATURES,
        "ucl_history_matches": len(ucl_history),
    }


# --------------------------------------------------
# FOOTBALL-DATA.ORG FIXTURES
# --------------------------------------------------

def fetch_competition_fixtures(competition_code: str):
    competition_code = competition_code.upper()

    if competition_code not in SUPPORTED_COMPETITIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported competition. Choose PL or CL.",
        )

    if not FOOTBALL_DATA_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Football data API token is missing from .env",
        )

    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=1)
    end_date = today + timedelta(days=14)

    params = {
        "dateFrom": start_date.isoformat(),
        "dateTo": end_date.isoformat(),
    }

    url = (
        f"{FOOTBALL_DATA_URL}/competitions/"
        f"{competition_code}/matches"
    )

    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}
    params = {
        "dateFrom": today.isoformat(),
        "dateTo": end_date.isoformat(),
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.HTTPError as exc:
        status_code = exc.response.status_code

        if status_code == 401:
            detail = (
                "Football data API rejected the token. "
                "Check your .env configuration."
            )
            raise HTTPException(status_code=502, detail=detail) from exc

        if status_code == 403:
            detail = (
                f"Your football-data.org API plan may not allow access "
                f"to {SUPPORTED_COMPETITIONS[competition_code]}."
            )
            raise HTTPException(status_code=502, detail=detail) from exc

        if status_code == 429:
            raise HTTPException(
                status_code=503,
                detail="Football data API rate limit reached. Try again later.",
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"Football data API returned HTTP {status_code}.",
        ) from exc

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the football data API.",
        ) from exc

    matches = []

    for match in data.get("matches", []):
        allowed_statuses = {
                    "SCHEDULED",
                    "TIMED",
                    "IN_PLAY",
                    "PAUSED",
                    "FINISHED",
                    "POSTPONED",
                    "SUSPENDED",
}

        if match.get("status") not in allowed_statuses:
            continue

        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}

        if not home_team.get("name") or not away_team.get("name"):
            continue

        matches.append({
            "id": match["id"],
            "competition_code": competition_code,
            "competition": (match.get("competition") or {}).get(
                "name", SUPPORTED_COMPETITIONS[competition_code]
            ),
            "home_team": home_team["name"],
            "away_team": away_team["name"],
            "kickoff": match["utcDate"],
            "status": match["status"],
            "matchday": match.get("matchday"),
        })

    matches.sort(key=lambda match: match["kickoff"])

    return {
        "competition_code": competition_code,
        "competition": SUPPORTED_COMPETITIONS[competition_code],
        "date_from": today.isoformat(),
        "date_to": end_date.isoformat(),
        "count": len(matches),
        "matches": matches,
        "score": match.get("score"),
        "home_score": (match.get("score") or {}).get("fullTime", {}).get("home"),
        "away_score": (match.get("score") or {}).get("fullTime", {}).get("away"),
    }


@app.get("/competitions")
def list_competitions():
    return {
        "competitions": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_COMPETITIONS.items()
        ]
    }


@app.get("/fixtures/upcoming")
def upcoming_fixtures(competition: str = "PL"):
    return fetch_competition_fixtures(competition)


def build_ucl_fixture_features(home_name: str, away_name: str, kickoff: str):
    """
    Build the exact 11 features used by the trained UCL model.

    Only completed historical matches strictly before the fixture's
    UTC calendar date are used. This prevents same-day result leakage.
    """
    if ucl_model is None or ucl_history.empty:
        return None, 0, 0

    kickoff_dt = pd.to_datetime(kickoff, errors="coerce", utc=True)

    if pd.isna(kickoff_dt):
        return None, 0, 0

    # The training process creates features for all matches on a date
    # before adding any results from that date to team history.
    history = ucl_history[ucl_history["date"] < kickoff_dt].copy()

    home_key = normalize_ucl_team_name(home_name)
    away_key = normalize_ucl_team_name(away_name)

    def team_recent(team_key):
        home_rows = history[history["home_key"] == team_key].copy()
        away_rows = history[history["away_key"] == team_key].copy()

        home_rows["gf"] = home_rows["home_goals"]
        home_rows["ga"] = home_rows["away_goals"]
        home_rows["points"] = home_rows["result"].map(
            {"H": 3, "D": 1, "A": 0}
        )
        home_rows["win"] = (home_rows["result"] == "H").astype(int)

        away_rows["gf"] = away_rows["away_goals"]
        away_rows["ga"] = away_rows["home_goals"]
        away_rows["points"] = away_rows["result"].map(
            {"A": 3, "D": 1, "H": 0}
        )
        away_rows["win"] = (away_rows["result"] == "A").astype(int)

        combined = pd.concat(
            [
                home_rows[["date", "gf", "ga", "points", "win"]],
                away_rows[["date", "gf", "ga", "points", "win"]],
            ],
            ignore_index=True,
        )

        return combined.sort_values("date").tail(5)

    home_recent = team_recent(home_key)
    away_recent = team_recent(away_key)

    def average(frame, column):
        return (
            float(frame[column].mean())
            if not frame.empty
            else np.nan
        )

    features = {
        "home_points_avg": average(home_recent, "points"),
        "home_goals_for_avg": average(home_recent, "gf"),
        "home_goals_against_avg": average(home_recent, "ga"),
        "home_win_rate": average(home_recent, "win"),
        "away_points_avg": average(away_recent, "points"),
        "away_goals_for_avg": average(away_recent, "gf"),
        "away_goals_against_avg": average(away_recent, "ga"),
        "away_win_rate": average(away_recent, "win"),
        "home_matches_played": len(home_recent),
        "away_matches_played": len(away_recent),
        "home_advantage": 1.0,
    }

    return features, len(home_recent), len(away_recent)


def predict_ucl_match(features: dict) -> dict:
    """Predict a Champions League fixture using the trained UCL model."""
    if ucl_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "The Champions League model is unavailable.",
                "reason": ucl_model_load_error
                or "Check models/footballiq_ucl_model.joblib and the UCL dataset.",
            },
        )

    input_data = pd.DataFrame(
        [[features.get(name, np.nan) for name in UCL_FEATURES]],
        columns=UCL_FEATURES,
    )

    try:
        predicted = str(ucl_model.predict(input_data)[0])
        probabilities = ucl_model.predict_proba(input_data)[0]

        classifier = (
            ucl_model[-1]
            if hasattr(ucl_model, "__getitem__")
            else ucl_model
        )

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                classifier.classes_, probabilities
            )
        }

    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate UCL prediction: {exc}",
        ) from exc

    return {
        "predicted_outcome": predicted,
        "probabilities": {
            "home_win": round(probability_map.get("H", 0.0), 4),
            "draw": round(probability_map.get("D", 0.0), 4),
            "away_win": round(probability_map.get("A", 0.0), 4),
        },
        "model_type": "LogisticRegression (UCL recent form)",
        "model_version": "footballiq_ucl_model.joblib",
        "status": "experimental",
        "note": (
            "Prediction generated by FootballIQ's Champions League-trained "
            "model using recent historical team form. Probabilities are "
            "experimental and are not guaranteed to be calibrated."
        ),
    }


# --------------------------------------------------
# FIXTURE PREDICTIONS + AUTOMATIC ODDS
# --------------------------------------------------

@app.get("/fixtures/predictions")
def fixture_predictions(competition: str = "PL"):
    fixture_data = fetch_competition_fixtures(competition)
    competition_code = fixture_data["competition_code"]

    # Odds are fetched for both competitions. For CL, we deliberately use
    # bookmaker-implied probabilities only: the saved ML models were trained
    # on Premier League data and are not validated for Champions League teams.
    try:
        odds_events = fetch_odds(competition_code)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        odds_events = []
        odds_warning = str(exc)
    else:
        odds_warning = None

    # Do not apply the Premier League form model to CL fixtures. When valid,
    # fixture-matched odds exist, return their normalized implied probabilities.
    # Champions League: use the trained UCL model, not bookmaker odds.
    if competition_code == "CL":
        if ucl_model is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Champions League predictions are unavailable.",
                    "reason": ucl_model_load_error
                    or "The UCL model could not be loaded.",
                },
            )

        predicted_matches = []

        for match in fixture_data["matches"]:
            features, home_count, away_count = build_ucl_fixture_features(
                home_name=match["home_team"],
                away_name=match["away_team"],
                kickoff=match["kickoff"],
            )

            if features is None:
                prediction = None
                unavailable_reason = (
                    "Could not build historical features for this fixture."
                )
                history_payload = {
                    "home_matches_available": home_count,
                    "away_matches_available": away_count,
                    "limited_history": True,
                }
            else:
                prediction = predict_ucl_match(features)
                unavailable_reason = None

                history_payload = {
                    "home_matches_available": home_count,
                    "away_matches_available": away_count,
                    "limited_history": (
                        home_count < 5 or away_count < 5
                    ),
                }

            predicted_matches.append({
                **match,
                "prediction": prediction,
                "history": history_payload,
                "team_form": None,
                "odds": None,
                "prediction_unavailable_reason": unavailable_reason,
            })

        return {
            **fixture_data,
            "matches": predicted_matches,
            "note": (
                "Champions League predictions use the trained "
                "footballiq_ucl_model.joblib model. Predictions are based "
                "on recent historical Champions League form and do not "
                "require bookmaker odds. Results are experimental."
            ),
        }

    # Premier League: retain the existing form + odds ML workflow.

    # Premier League: retain the existing form + odds ML workflow.
    try:
        history = load_historical_history()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predicted_matches = []

    for match in fixture_data["matches"]:
        features, home_history_count, away_history_count = (
            build_fixture_features(
                history=history,
                home_name=match["home_team"],
                away_name=match["away_team"],
                kickoff=match["kickoff"],
            )
        )

        matched_odds = find_matching_odds(match, odds_events)

        if matched_odds:
            try:
                prediction = predict_with_form_and_odds(
                    features,
                    matched_odds,
                )
            except (ValueError, KeyError, TypeError):
                # A malformed odds entry should not break the whole endpoint.
                matched_odds = None
                prediction = predict_form_only(features)
        else:
            prediction = predict_form_only(features)

        prediction["odds"] = (
            {
                "home": matched_odds["home_decimal_odds"],
                "draw": matched_odds["draw_decimal_odds"],
                "away": matched_odds["away_decimal_odds"],
                "source": matched_odds.get("source", "TheRundown"),
                "kickoff": matched_odds.get("kickoff"),
                "delay_minutes": matched_odds.get("delay_minutes"),
            }
            if matched_odds
            else None
        )

        home_form = build_team_form(
            history,
            match["home_team"],
            match["kickoff"],
        )

        away_form = build_team_form(
            history,
            match["away_team"],
            match["kickoff"],
        )

        home_venue_form = build_team_form(
            history,
            match["home_team"],
            match["kickoff"],
            venue="home",
        )

        away_venue_form = build_team_form(
            history,
            match["away_team"],
            match["kickoff"],
            venue="away",
        )

        predicted_matches.append({
            **match,
            "prediction": prediction,
            "history": {
                "home_matches_available": home_history_count,
                "away_matches_available": away_history_count,
                "limited_history": (
                    home_history_count < 5
                    or away_history_count < 5
                ),
            },
            "team_form": {
                "home": home_form,
                "away": away_form,
                "home_venue": home_venue_form,
                "away_venue": away_venue_form,
            },
        })

    response = {
        **fixture_data,
        "matches": predicted_matches,
        "note": (
            "Experimental Premier League predictions. Combined form + odds "
            "predictions are used only when fixture-matched bookmaker odds "
            "are available; otherwise the form-only model is used."
        ),
    }

    if odds_warning:
        response["odds_warning"] = (
            "The odds provider could not be used for this request. "
            "Form-only predictions were returned where possible."
        )

    return response


# --------------------------------------------------
# MANUAL PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_match(request: PredictionRequest):
    return predict_form_only(request.features)


# --------------------------------------------------
# MANUAL FORM + ODDS PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict-form-odds")
def predict_form_odds(request: FormOddsPredictionRequest):
    supplied = request.features

    missing = [
        name for name in FORM_FEATURE_NAMES
        if name not in supplied
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_form_features": missing},
        )

    try:
        market = decimal_odds_to_probabilities(
            request.home_odds,
            request.draw_odds,
            request.away_odds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    combined_features = {
        **{name: supplied[name] for name in FORM_FEATURE_NAMES},
        **market,
    }

    missing_combined = [
        name for name in FORM_ODDS_FEATURES
        if name not in combined_features
    ]

    if missing_combined:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "The saved combined model expects unexpected features.",
                "missing_features": missing_combined,
            },
        )

    input_data = pd.DataFrame(
        [[combined_features[name] for name in FORM_ODDS_FEATURES]],
        columns=FORM_ODDS_FEATURES,
    )

    probabilities = form_odds_model.predict_proba(input_data)[0]
    classifier = (
        form_odds_model[-1]
        if hasattr(form_odds_model, "__getitem__")
        else form_odds_model
    )

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(
            classifier.classes_,
            probabilities,
        )
    }

    return {
        "predicted_outcome": str(form_odds_model.predict(input_data)[0]),
        "probabilities": {
            "home_win": round(probability_map.get("H", 0.0), 4),
            "draw": round(probability_map.get("D", 0.0), 4),
            "away_win": round(probability_map.get("A", 0.0), 4),
        },
        "market_probabilities": {
            "home_win": round(market["market_home_prob"], 4),
            "draw": round(market["market_draw_prob"], 4),
            "away_win": round(market["market_away_prob"], 4),
        },
        "model_type": "LogisticRegression (form + bookmaker odds)",
        "features_used": len(FORM_ODDS_FEATURES),
        "status": "experimental",
        "note": (
            "Experimental predictions. Probabilities are not guaranteed "
            "to be calibrated and are not betting advice."
        ),
    }


# --------------------------------------------------
# AUTOMATIC ODDS ENDPOINT
# --------------------------------------------------


@app.get("/odds/upcoming")
def upcoming_odds(competition: str = "PL"):
    competition_code = competition.upper()

    if competition_code not in SUPPORTED_COMPETITIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported competition. Choose PL or CL.",
        )

    try:
        events = fetch_odds(competition_code)
        return {
            "competition": competition_code,
            "events": events,
            "count": len(events),
            "source": "TheRundown",
        }

    except Exception as exc:
        # Log the real error in the Uvicorn terminal for debugging.
        print(
            f"TheRundown error for {competition_code}: "
            f"{type(exc).__name__}: {exc}"
        )

        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if isinstance(exc, requests.RequestException):
            raise HTTPException(
                status_code=502,
                detail=f"TheRundown request failed: {exc}",
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"TheRundown integration failed: {type(exc).__name__}: {exc}",
        ) from exc

