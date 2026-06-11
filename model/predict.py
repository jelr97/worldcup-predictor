"""Per-match prediction pipeline: odds -> constraints -> rates -> picks."""
from dataclasses import dataclass, field

from data import elo as elo_mod
from data.team_names import same_team
from model import markets, picks, poisson


@dataclass
class MatchPrediction:
    fixture: dict
    source: str                    # 'market' | 'elo' | 'none'
    probs: dict | None = None      # consensus 1X2 in fixture orientation
    lam_home: float | None = None
    lam_away: float | None = None
    pool1: dict | None = None
    pool2: dict | None = None
    ep_table: list = field(default_factory=list)
    books_count: int = 0
    odds_age_hours: float | None = None
    elo_disagrees: bool = False
    note: str = ""


def match_event(fixture, events):
    """Find the odds event for a fixture. Returns (event, swapped)."""
    for e in events or []:
        if same_team(fixture["home"], e["home_team"]) and \
           same_team(fixture["away"], e["away_team"]):
            return e, False
        if same_team(fixture["home"], e["away_team"]) and \
           same_team(fixture["away"], e["home_team"]):
            return e, True
    return None, False


def _finish(pred, constraints, cfg):
    lh, la = poisson.solve_rates(constraints)
    m = poisson.score_matrix(lh, la)
    p = picks.top_picks(m, cfg["pool"]["pts_exact"], cfg["pool"]["pts_outcome"])
    pred.lam_home, pred.lam_away = lh, la
    pred.probs = constraints["1x2"]
    pred.pool1, pred.pool2, pred.ep_table = p["pool1"], p["pool2"], p["table"]
    return pred


def _elo_1x2_for(fixture, elo_ratings):
    rh = elo_mod.rating_for(elo_ratings, fixture["home"])
    ra = elo_mod.rating_for(elo_ratings, fixture["away"])
    if rh is None or ra is None:
        return None
    return elo_mod.elo_1x2(elo_mod.win_expectancy(rh, ra))


def predict_match(fixture, event, extras, swapped, elo_ratings, cfg, odds_age=None):
    pred = MatchPrediction(fixture=fixture, source="none", odds_age_hours=odds_age)
    constraints = markets.build_constraints(event, extras) if event else None
    if constraints:
        if swapped:  # totals/btts are symmetric; only 1X2 needs flipping
            constraints["1x2"] = {"home": constraints["1x2"]["away"],
                                  "draw": constraints["1x2"]["draw"],
                                  "away": constraints["1x2"]["home"]}
        pred.source = "market"
        pred.books_count = constraints.get("books_count", 0)
        _finish(pred, constraints, cfg)
        elo_probs = _elo_1x2_for(fixture, elo_ratings)
        if elo_probs:
            thr = cfg["elo"]["disagreement_threshold"]
            pred.elo_disagrees = abs(elo_probs["home"] - pred.probs["home"]) > thr
        return pred
    elo_probs = _elo_1x2_for(fixture, elo_ratings)
    if elo_probs:
        pred.source = "elo"
        pred.note = "model-only (no market odds)"
        return _finish(pred, {"1x2": elo_probs, "totals": [], "btts": None}, cfg)
    pred.note = "no odds and no Elo rating - no pick"
    return pred


def predict_upcoming(fixtures_window, events, extras_by_event_id, elo_ratings,
                     cfg, odds_age=None):
    preds = []
    for f in fixtures_window:
        e, swapped = match_event(f, events)
        extras = (extras_by_event_id or {}).get(e["id"]) if e else None
        preds.append(predict_match(f, e, extras, swapped, elo_ratings, cfg, odds_age))
    return preds
