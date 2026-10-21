import re
from pathlib import Path

import pandas as pd


# FootballIQ project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "ucl" / "source"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "ucl_matches.csv"


MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

WEEKDAYS = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"

DATE_PATTERN = re.compile(
    rf"^\s*{WEEKDAYS}\s+([A-Z][a-z]{{2}})\s+(\d{{1,2}})"
    rf"(?:\s+(\d{{4}}))?\s*$"
)

# Supports match lines with or without a kickoff time.
# Example:
# 18:45 BSC Young Boys (SUI) v Aston Villa FC (ENG) 0-3 (0-2)
MATCH_PATTERN = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(.+?)\s+v\s+(.+?)\s+"
    r"(\d+)\s*-\s*(\d+)"
    r"(?:\s+\((\d+)\s*-\s*(\d+)\))?"
    r"(?:\s+.*)?$"
)

# Handles matches decided after extra time and penalties.
PENALTY_PATTERN = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(.+?)\s+v\s+(.+?)\s+"
    r"(\d+)\s*-\s*(\d+)\s+pen\.\s+"
    r"(\d+)\s*-\s*(\d+)\s+a\.e\.t\.\s+"
    r"\(\s*(\d+)\s*-\s*(\d+)\s*,\s*"
    r"(\d+)\s*-\s*(\d+)\s*\)"
)


def get_season_start_year(season_folder: str) -> int:
    """Convert a folder such as 2024-25 into its starting year."""
    match = re.fullmatch(r"(\d{4})-\d{2}", season_folder)

    if not match:
        raise ValueError(f"Unexpected season folder: {season_folder}")

    return int(match.group(1))


def parse_season_file(file_path: Path) -> list[dict]:
    """Parse completed Champions League matches from one cl.txt file."""
    season = file_path.parent.name
    season_start_year = get_season_start_year(season)

    current_date = None
    current_stage = None
    matches = []

    with file_path.open("r", encoding="utf-8-sig") as file:
        for line_number, original_line in enumerate(file, start=1):
            line = original_line.rstrip()
            stripped = line.strip()

            # Stage headings, for example: League, Matchday 1
            if stripped.startswith("▪"):
                current_stage = stripped.lstrip("▪").strip()
                continue

            # Date headings may omit the year after the first date.
            date_match = DATE_PATTERN.match(line)

            if date_match:
                month_name, day_text, explicit_year = date_match.groups()
                month = MONTHS.get(month_name)

                if month is None:
                    continue

                day = int(day_text)

                if explicit_year:
                    year = int(explicit_year)
                else:
                    # European football seasons cross calendar years.
                    year = (
                        season_start_year
                        if month >= 7
                        else season_start_year + 1
                    )

                try:
                    current_date = pd.Timestamp(
                        year=year,
                        month=month,
                        day=day,
                    )
                except ValueError:
                    print(
                        f"Skipping invalid date at "
                        f"{file_path}:{line_number}: {line}"
                    )
                    current_date = None

                continue

            # Ignore lines before the first valid date.
            if current_date is None:
                continue

            # First check for penalty shootout match lines.
            penalty_match = PENALTY_PATTERN.match(line)

            if penalty_match:
                (
                    home_team,
                    away_team,
                    _penalty_home,
                    _penalty_away,
                    _extra_time_home,
                    _extra_time_away,
                    _halftime_home,
                    _halftime_away,
                    regular_home,
                    regular_away,
                ) = penalty_match.groups()

                # The prediction target is the result after regular time,
                # not the extra-time or penalty shootout result.
                home_goals = int(regular_home)
                away_goals = int(regular_away)

                halftime_home = None
                halftime_away = None

            else:
                match = MATCH_PATTERN.match(line)

                if not match:
                    continue

                (
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    halftime_home,
                    halftime_away,
                ) = match.groups()

                home_goals = int(home_goals)
                away_goals = int(away_goals)

                halftime_home = (
                    int(halftime_home)
                    if halftime_home is not None
                    else None
                )
                halftime_away = (
                    int(halftime_away)
                    if halftime_away is not None
                    else None
                )

            # Determine the regular-time result.
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            matches.append(
                {
                    "date": current_date,
                    "season": season,
                    "stage": current_stage,
                    "home_team": home_team.strip(),
                    "away_team": away_team.strip(),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "halftime_home_goals": halftime_home,
                    "halftime_away_goals": halftime_away,
                    "source_file": str(
                        file_path.relative_to(SOURCE_DIR)
                    ),
                }
            )

    return matches


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"UCL source directory not found: {SOURCE_DIR}"
        )

    # Only cl.txt: exclude other competitions and unrelated files.
    season_files = sorted(SOURCE_DIR.glob("*/cl.txt"))

    if not season_files:
        raise FileNotFoundError(
            f"No Champions League cl.txt files found in {SOURCE_DIR}"
        )

    all_matches = []

    for file_path in season_files:
        season_matches = parse_season_file(file_path)
        all_matches.extend(season_matches)

        print(
            f"{file_path.parent.name}: "
            f"{len(season_matches)} parsed matches"
        )

    if not all_matches:
        raise ValueError(
            "No matches were parsed. Check the source file format."
        )

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"])

    # Stable chronological ordering for later model training.
    df = df.sort_values(
        ["date", "season", "home_team", "away_team"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n========== UCL DATA SUMMARY ==========")
    print(f"Seasons found: {len(season_files)}")
    print(f"Parsed matches: {len(df)}")
    print(
        f"Date range: {df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )
    print(f"Unique home teams: {df['home_team'].nunique()}")
    print(f"Unique away teams: {df['away_team'].nunique()}")

    print("\nMatch outcomes:")
    print(
        df["result"]
        .value_counts()
        .reindex(["H", "D", "A"], fill_value=0)
    )

    print("\nMatches per season:")
    print(df.groupby("season").size().to_string())

    print(f"\nSaved CSV to: {OUTPUT_FILE}")

    print("\nFirst five rows:")
    print(
        df[
            [
                "date",
                "season",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
