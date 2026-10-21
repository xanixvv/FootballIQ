
from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"


HISTORY_FILES = [
    "E0_2020_21.csv",
    "E0_2021_22.csv",
    "E0_2022_23.csv",
    "E0_2023_24.csv",
    "E0_2024_25.csv",
    "E0.csv",
]



FEATURES = [
    # Overall form: last 5 matches
    "home_points_avg",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_points_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",

    # Venue-specific form
    "home_home_points_avg",
    "home_home_goals_for_avg",
    "home_home_goals_against_avg",
    "away_away_points_avg",
    "away_away_goals_for_avg",
    "away_away_goals_against_avg",

    # Differences between the teams
    "points_difference",
    "goal_difference_difference",

    # Longer-term form: last 10 matches
    "home_points_avg_10",
    "home_goal_difference_avg_10",
    "away_points_avg_10",
    "away_goal_difference_avg_10",
]


TEAM_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Coventry City FC": "Coventry City",
}


def normalize_team(name):
    """Normalize common API team names to historical CSV team names."""
    name = str(name).strip()

    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]

    for suffix in (" FC", " AFC"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


def _normalise_date(value):
    """Convert a date or timestamp to a timezone-naive pandas timestamp."""
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)

    return timestamp


@lru_cache(maxsize=1)
def load_historical_history():
    """
    Load historical PL results into per-team histories.

    Each history entry is:
        (date, points, goals_for, goals_against, venue)

    Venue is 'home' or 'away'.
    """
    history = {}
    seen_matches = set()

    required = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]

    for filename in HISTORY_FILES:
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Historical CSV is missing: {path}"
            )

        df = pd.read_csv(path)

        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{filename} is missing required columns: {missing}"
            )

        df["DateParsed"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce",
        )
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

        df = df.dropna(
            subset=[
                "DateParsed",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
            ]
        )

        df = df[
            df["FTR"].isin(["H", "D", "A"])
            & (df["HomeTeam"] != "")
            & (df["AwayTeam"] != "")
        ]

        sort_columns = ["DateParsed"]
        if "Time" in df.columns:
            sort_columns.append("Time")

        df = df.sort_values(sort_columns)

        for _, match in df.iterrows():
            match_date = _normalise_date(match["DateParsed"])
            home = normalize_team(match["HomeTeam"])
            away = normalize_team(match["AwayTeam"])
            home_goals = int(match["FTHG"])
            away_goals = int(match["FTAG"])

            # Deduplicate repeated rows across overlapping source files.
            # Include the date, teams and score so different matches remain.
            dedup_key = (
                match_date.normalize(),
                home,
                away,
                home_goals,
                away_goals,
            )

            if dedup_key in seen_matches:
                continue

            seen_matches.add(dedup_key)

            if home_goals > away_goals:
                home_points, away_points = 3, 0
                calculated_result = "H"
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
                calculated_result = "A"
            else:
                home_points, away_points = 1, 1
                calculated_result = "D"

            # Use the goals to determine the result so inconsistent source
            # result labels cannot corrupt the points calculation.
            _ = calculated_result

            history.setdefault(home, []).append(
                (
                    match_date,
                    home_points,
                    home_goals,
                    away_goals,
                    "home",
                )
            )

            history.setdefault(away, []).append(
                (
                    match_date,
                    away_points,
                    away_goals,
                    home_goals,
                    "away",
                )
            )

    for team in history:
        history[team].sort(key=lambda entry: entry[0])

    return history


def _matches_before_date(history, team, kickoff):
    """Return matches strictly before the kickoff's calendar date."""
    fixture_date = _normalise_date(kickoff).normalize()

    return [
        match
        for match in history.get(team, [])
        if _normalise_date(match[0]).normalize() < fixture_date
    ]


def summarize_form(matches, venue=None):
    """Calculate average points, goals scored and goals conceded."""
    if venue is not None:
        matches = [
            match for match in matches
            if match[4] == venue
        ]

    # Histories are sorted chronologically; only the last five are used.
    recent = matches[-5:]

    # Preserve the defaults used by the existing model.
    if not recent:
        return 1.0, 1.3, 1.3

    points_avg = sum(match[1] for match in recent) / len(recent)
    goals_for_avg = sum(match[2] for match in recent) / len(recent)
    goals_against_avg = sum(match[3] for match in recent) / len(recent)

    return points_avg, goals_for_avg, goals_against_avg



def build_fixture_features(history, home_name, away_name, kickoff):
    """Build pre-match features using each team's previous results."""
    home = normalize_team(home_name)
    away = normalize_team(away_name)

    home_history = _matches_before_date(history, home, kickoff)
    away_history = _matches_before_date(history, away, kickoff)

    # Existing five-match features
    home_form = summarize_form(home_history)
    away_form = summarize_form(away_history)

    home_home_form = summarize_form(home_history, venue="home")
    away_away_form = summarize_form(away_history, venue="away")

    home_goal_difference = home_form[1] - home_form[2]
    away_goal_difference = away_form[1] - away_form[2]

    # New ten-match features
    home_recent_10 = home_history[-10:]
    away_recent_10 = away_history[-10:]

    def averages(matches):
        if not matches:
            return 1.0, 0.0

        points_avg = sum(match[1] for match in matches) / len(matches)
        goal_difference_avg = sum(
            match[2] - match[3] for match in matches
        ) / len(matches)

        return points_avg, goal_difference_avg

    home_points_10, home_gd_10 = averages(home_recent_10)
    away_points_10, away_gd_10 = averages(away_recent_10)

    features = {
        "home_points_avg": home_form[0],
        "home_goals_for_avg": home_form[1],
        "home_goals_against_avg": home_form[2],
        "away_points_avg": away_form[0],
        "away_goals_for_avg": away_form[1],
        "away_goals_against_avg": away_form[2],
        "home_home_points_avg": home_home_form[0],
        "home_home_goals_for_avg": home_home_form[1],
        "home_home_goals_against_avg": home_home_form[2],
        "away_away_points_avg": away_away_form[0],
        "away_away_goals_for_avg": away_away_form[1],
        "away_away_goals_against_avg": away_away_form[2],
        "points_difference": home_form[0] - away_form[0],
        "goal_difference_difference": (
            home_goal_difference - away_goal_difference
        ),
        "home_points_avg_10": home_points_10,
        "home_goal_difference_avg_10": home_gd_10,
        "away_points_avg_10": away_points_10,
        "away_goal_difference_avg_10": away_gd_10,
    }

    features = {name: features[name] for name in FEATURES}

    return features, len(home_history), len(away_history)


def build_team_form(history, team_name, kickoff, venue=None):
    """Summarize a team's last five completed matches before kickoff."""
    team = normalize_team(team_name)
    matches = _matches_before_date(history, team, kickoff)

    if venue is not None:
        matches = [
            match for match in matches
            if match[4] == venue
        ]

    recent = matches[-5:][::-1]

    results = []

    for match in recent:
        points = match[1]

        if points == 3:
            results.append("W")
        elif points == 1:
            results.append("D")
        else:
            results.append("L")

    count = len(recent)

    return {
        "team": team_name,
        "matches_played": count,
        "form": results,
        "points": sum(match[1] for match in recent),
        "average_goals_scored": (
            round(sum(match[2] for match in recent) / count, 2)
            if count else None
        ),
        "average_goals_conceded": (
            round(sum(match[3] for match in recent) / count, 2)
            if count else None
        ),
        "home_matches": sum(
            1 for match in recent if match[4] == "home"
        ),
        "away_matches": sum(
            1 for match in recent if match[4] == "away"
        ),
        "venue_filter": venue,
    }
