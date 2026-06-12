"""Load and query expert score predictions from data/experts.csv.

CSV columns: group, team_a, team_b, davo_a, davo_b, maldini_a, maldini_b
"""
import csv
from pathlib import Path

from data.team_names import same_team

_CSV_PATH = Path(__file__).parent / "experts.csv"


def load_experts() -> dict:
    """Return a dict keyed by (normalized_a, normalized_b) -> row dict.

    Returns an empty dict if the CSV is missing or corrupt.
    Key order is canonical (team_a, team_b) as stored in CSV.
    """
    if not _CSV_PATH.exists():
        return {}
    try:
        experts = {}
        with open(_CSV_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["team_a"], row["team_b"])
                experts[key] = {
                    "group": row["group"],
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "davo_a": int(row["davo_a"]),
                    "davo_b": int(row["davo_b"]),
                    "maldini_a": int(row["maldini_a"]),
                    "maldini_b": int(row["maldini_b"]),
                }
        return experts
    except Exception:
        return {}


def picks_for(experts: dict, home: str, away: str) -> dict | None:
    """Return expert pick dict for a fixture, handling either orientation.

    When the row is stored as (away, home), the score tuples are swapped so
    that the returned values are always in the caller's (home, away) frame.

    Returns None if no expert data exists for this fixture.
    """
    for (ta, tb), row in experts.items():
        if same_team(home, ta) and same_team(away, tb):
            # Canonical orientation: team_a == home
            return {
                "davo": (row["davo_a"], row["davo_b"]),
                "maldini": (row["maldini_a"], row["maldini_b"]),
            }
        if same_team(home, tb) and same_team(away, ta):
            # Reversed orientation: stored as (away, home) — swap scores
            return {
                "davo": (row["davo_b"], row["davo_a"]),
                "maldini": (row["maldini_b"], row["maldini_a"]),
            }
    return None
