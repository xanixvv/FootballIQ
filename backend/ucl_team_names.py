"""Shared team-name canonicalization for UCL training and API inference."""
import re
import unicodedata


def normalize_ucl_team_name(name: str) -> str:
    """Return a stable key for common historical/provider name variations."""
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().strip()
    # Some football-data sources append country codes, e.g. Arsenal (ENG).
    value = re.sub(r"\s*\([a-z]{3}\)\s*$", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    # Remove organizational suffixes, not meaningful words such as United or City.
    suffixes = {"fc", "cf", "sc", "afc", "ac", "as", "fk", "kv", "club", "sv", "sk", "pae", "ssc"}
    tokens = [token for token in value.split() if token not in suffixes]
    value = " ".join(tokens)
    aliases = {
        "man city": "manchester city",
        "manchester city": "manchester city",
        "man utd": "manchester united",
        "manchester utd": "manchester united",
        "manchester united": "manchester united",
        "paris saint germain": "paris saint germain",
        "psg": "paris saint germain",
        "internazionale": "inter milan",
        "inter milano": "inter milan",
        "inter milan": "inter milan",
        "inter": "inter milan",
        "bayern munchen": "bayern munich",
        "bayern munich": "bayern munich",
        "atletico de madrid": "atletico madrid",
        "atletico madrid": "atletico madrid",
        "sporting clube de portugal": "sporting cp",
        "sporting lisbon": "sporting cp",
        "sporting cp": "sporting cp",
        "red bull salzburg": "salzburg",
        "fc salzburg": "salzburg",
        "salzburg": "salzburg",
        "borussia monchengladbach": "borussia monchengladbach",
        "bodo glimt": "bodo glimt",
        "bodo glimt fk": "bodo glimt",
        "young boys": "young boys",
        "bsc young boys": "young boys",
        "shakhtar donetsk": "shakhtar donetsk",
        "fc shakhtar donetsk": "shakhtar donetsk",
        "slovan bratislava": "slovan bratislava",
        "sk slovan bratislava": "slovan bratislava",
        "sab ah fk": "sabah",
        "sabah fk": "sabah",
        "sabah": "sabah",
        "fenerbahce": "fenerbahce",
        "fenerbahce istanbul": "fenerbahce",
        "vfb stuttgart": "stuttgart",
        "stuttgart": "stuttgart",
        "as roma": "roma",
        "roma": "roma",
        "psv eindhoven": "psv",
        "psv": "psv",
        "villarreal": "villarreal",
        "villarreal cf": "villarreal",
        "lask linz": "lask linz",
        "aek athen": "aek",
        "aek athens": "aek",
        "pae aek": "aek",
    }
    return aliases.get(value, value)
