"""World Football Elo ratings (bundled snapshot, refreshed via scripts/snapshot_elo.py)."""
import csv
from pathlib import Path

from data.team_names import normalize

SNAPSHOT = Path(__file__).parent / "elo_snapshot.csv"


def load_ratings(path=SNAPSHOT):
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {normalize(r["team"]): float(r["rating"]) for r in csv.DictReader(f)}


def rating_for(ratings, team):
    return ratings.get(normalize(team))


def win_expectancy(r_home, r_away):
    """Elo expectancy (draws count half). No home advantage: WC is ~neutral."""
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))


def elo_1x2(we):
    """Split Elo expectancy into 1X2: we = p_home + p_draw/2.

    Draw probability shrinks linearly as the matchup gets lopsided
    (balanced ~27%, floor 5%).
    """
    p_draw = max(0.05, 0.27 - 0.22 * abs(2 * we - 1))
    p_home = max(0.01, we - p_draw / 2)
    p_away = max(0.01, 1 - p_home - p_draw)
    total = p_home + p_draw + p_away
    return {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}
