
import pandas as pd

from backend.form_features import (
    FEATURES,
    HISTORY_FILES,
    RAW_DIR,
    load_historical_history,
    build_fixture_features,
)

history = load_historical_history()

print("FootballIQ feature-history audit")
print("=" * 50)

print("\nHistory size:", sum(len(v) for v in history.values()))
print("Teams in history:", len(history))

for filename in HISTORY_FILES:
    path = RAW_DIR / filename
    df = pd.read_csv(path)

    df["parsed_date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )

    print(
        f"{filename}: rows={len(df)}, "
        f"valid_dates={df['parsed_date'].notna().sum()}, "
        f"date_range={df['parsed_date'].min()} "
        f"to {df['parsed_date'].max()}"
    )

print("\nFeature smoke test:")

features, home_count, away_count = build_fixture_features(
    history=history,
    home_name="Arsenal",
    away_name="Chelsea",
    kickoff="2024-12-01",
)

print("Feature count:", len(features))
print("Expected feature count:", len(FEATURES))
print("All features present:", set(features) == set(FEATURES))
print("Home matches available before date:", home_count)
print("Away matches available before date:", away_count)
print("Features:", features)

assert len(features) == len(FEATURES)
assert all(pd.notna(value) for value in features.values())

print("\nAudit completed.")
