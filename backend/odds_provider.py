
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://therundown.io/api/v2"
API_KEY = os.getenv("THERUNDOWN_API_KEY")

COMPETITIONS = {"PL": 11, "CL": 16}
AFFILIATE_IDS = "19,23"


def american_to_decimal(price):
    if price is None or price == 0:
        return None

    price = float(price)
    if price > 0:
        return round(1 + price / 100, 3)

    return round(1 + 100 / abs(price), 3)


def fetch_odds(competition: str):
    if not API_KEY:
        raise RuntimeError("THERUNDOWN_API_KEY is missing from .env")

    competition = competition.upper()
    sport_id = COMPETITIONS.get(competition)
    if sport_id is None:
        raise ValueError("Competition must be PL or CL")

    # Fetch upcoming dates from the provider.
    dates_response = requests.get(
        f"{API_URL}/sports/dates",
        headers={"X-TheRundown-Key": API_KEY},
        params={
            "sport_ids": str(sport_id),
            "format": "epoch",
            "exclude_canceled": "true",
        },
        timeout=20,
    )
    dates_response.raise_for_status()

    dates = dates_response.json().get(
        str(sport_id), {}
    ).get("dates", [])

    now = datetime.now(timezone.utc).timestamp()
    future_dates = sorted(
        float(stamp) for stamp in dates if float(stamp) > now
    )

    results = []

    for stamp in future_dates[:5]:
        date = datetime.fromtimestamp(
            stamp, timezone.utc
        ).date().isoformat()

        response = requests.get(
            f"{API_URL}/sports/{sport_id}/events/{date}",
            headers={"X-TheRundown-Key": API_KEY},
            params={
                "market_ids": "1",
                "affiliate_ids": AFFILIATE_IDS,
                "main_line": "true",
                "hide_closed": "true",
            },
            timeout=20,
        )
        response.raise_for_status()

        for event in response.json().get("events", []):
            teams = event.get("teams", [])
            home_team = next(
                (t.get("name") for t in teams if t.get("is_home")),
                None,
            )
            away_team = next(
                (t.get("name") for t in teams if t.get("is_away")),
                None,
            )

            market = next(
                (
                    m for m in event.get("markets", [])
                    if m.get("market_id") == 1
                    and m.get("period_id") == 0
                ),
                None,
            )
            if not market:
                continue

            # Keep the freshest available price per outcome.
            outcomes = {}
            for participant in market.get("participants", []):
                name = participant.get("name", "").strip().lower()
                if name == "draw":
                    key = "draw"
                elif home_team and name == home_team.lower():
                    key = "home"
                elif away_team and name == away_team.lower():
                    key = "away"
                else:
                    continue

                for line in participant.get("lines", []):
                    prices = line.get("prices", {})
                    candidates = [
                        price_data
                        for affiliate, price_data in prices.items()
                        if affiliate in AFFILIATE_IDS.split(",")
                        and price_data.get("price") not in (None, 0)
                    ]
                    if not candidates:
                        continue

                    latest = max(
                        candidates,
                        key=lambda p: p.get("updated_at", ""),
                    )
                    current = outcomes.get(key)
                    if (
                        current is None
                        or latest.get("updated_at", "") > current["updated_at"]
                    ):
                        decimal = american_to_decimal(latest["price"])
                        outcomes[key] = {
                            "decimal_odds": decimal,
                            "updated_at": latest.get("updated_at"),
                            "affiliate_id": next(
                                affiliate
                                for affiliate, p in prices.items()
                                if p is latest
                            ),
                        }

            if not all(k in outcomes for k in ("home", "draw", "away")):
                continue

            results.append({
                "event_id": event.get("event_id"),
                "competition": competition,
                "kickoff": event.get("event_date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_decimal_odds": outcomes["home"]["decimal_odds"],
                "draw_decimal_odds": outcomes["draw"]["decimal_odds"],
                "away_decimal_odds": outcomes["away"]["decimal_odds"],
                "source": "TheRundown",
                "delay_minutes": 5,
            })

    return results
