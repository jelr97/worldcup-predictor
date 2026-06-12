"""Per-match prediction pipeline: odds -> constraints -> rates -> picks."""
from dataclasses import dataclass, field

import numpy as np

from data import elo as elo_mod
from data.experts import picks_for
from data.team_names import same_team
from model import markets, picks, poisson
from model.experts import experts_matrix
from model.poisson import outcome_probs, score_matrix, solve_rates


@dataclass
class MatchPrediction:
    fixture: dict
    source: str                    # 'market' | 'market+experts' | 'experts' | 'elo' | 'none'
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
    members: list = field(default_factory=list)   # e.g. ['market', 'experts']
    expert_picks: dict | None = None              # {'davo': '2-1', 'maldini': '2-0'}


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


def stage_points(stage, cfg):
    """Map a fixture stage label to its scoring tier points."""
    s = cfg["pool"]["scoring"]
    if stage.startswith("Group"):
        return s["group"]
    try:
        rnd = int(stage.rsplit("R", 1)[1])   # "Knockout R4" -> 4
    except (IndexError, ValueError):
        return s["qf_plus"]
    return s["r32_r16"] if rnd <= 5 else s["qf_plus"]


def _matrix_to_picks(m, pts):
    """Derive picks and probs from a blended score matrix."""
    return picks.top_picks(m, pts)


def _elo_1x2_for(fixture, elo_ratings):
    rh = elo_mod.rating_for(elo_ratings, fixture["home"])
    ra = elo_mod.rating_for(elo_ratings, fixture["away"])
    if rh is None or ra is None:
        return None
    return elo_mod.elo_1x2(elo_mod.win_expectancy(rh, ra))


def _finish_from_matrix(pred, m, pts):
    """Populate pred from a pre-built blended score matrix."""
    p = picks.top_picks(m, pts)
    pred.probs = outcome_probs(m)
    pred.pool1, pred.pool2, pred.ep_table = p["pool1"], p["pool2"], p["table"]
    return pred


def _finish(pred, constraints, cfg):
    lh, la = solve_rates(constraints)
    m = score_matrix(lh, la)
    pts = stage_points(pred.fixture["stage"], cfg)
    p = picks.top_picks(m, pts)
    pred.lam_home, pred.lam_away = lh, la
    pred.probs = constraints["1x2"]
    pred.pool1, pred.pool2, pred.ep_table = p["pool1"], p["pool2"], p["table"]
    return pred


def predict_match(fixture, event, extras, swapped, elo_ratings, cfg,
                  odds_age=None, experts=None):
    pred = MatchPrediction(fixture=fixture, source="none", odds_age_hours=odds_age)
    pts = stage_points(fixture["stage"], cfg)
    ensemble_cfg = cfg.get("ensemble", {"market": 1.0, "experts": 1.0})

    # ── market matrix ─────────────────────────────────────────────────────────
    market_matrix = None
    market_constraints = None
    if event:
        constraints = markets.build_constraints(event, extras)
        if constraints:
            if swapped:
                constraints["1x2"] = {
                    "home": constraints["1x2"]["away"],
                    "draw": constraints["1x2"]["draw"],
                    "away": constraints["1x2"]["home"],
                }
                # When event is swapped, the spread lines were computed from
                # the event's home perspective. Fixture-home = event-away, so
                # fixture-home handicap = -(event-home handicap) AND the cover
                # probability must be complemented: P(fixture-home covers L) =
                # 1 - P(event-home covers -L).
                constraints["spreads"] = [
                    (-line, 1.0 - p) for line, p in (constraints.get("spreads") or [])
                ]
            market_constraints = constraints
            lh, la = solve_rates(constraints)
            pred.lam_home, pred.lam_away = lh, la
            pred.books_count = constraints.get("books_count", 0)
            market_matrix = score_matrix(lh, la)

    # ── experts matrix ────────────────────────────────────────────────────────
    exp_matrix = None
    raw_picks = None
    if experts:
        raw_picks = picks_for(experts, fixture["home"], fixture["away"])
        if raw_picks is not None:
            exp_matrix = experts_matrix(raw_picks)

    # ── collect available members ─────────────────────────────────────────────
    member_matrices = {}
    if market_matrix is not None:
        member_matrices["market"] = market_matrix
    if exp_matrix is not None:
        member_matrices["experts"] = exp_matrix

    # ── Elo fallback: only when no market AND no experts ─────────────────────
    if not member_matrices:
        elo_probs = _elo_1x2_for(fixture, elo_ratings)
        if elo_probs:
            pred.source = "elo"
            pred.note = "model-only (no market odds)"
            pred.members = ["elo"]
            return _finish(pred, {"1x2": elo_probs, "totals": [], "btts": None}, cfg)
        pred.note = "no odds and no Elo rating - no pick"
        return pred

    # ── blend available members using config weights ──────────────────────────
    total_weight = 0.0
    blended = None
    member_names = []
    for name, mat in member_matrices.items():
        w = ensemble_cfg.get(name, 1.0)
        if blended is None:
            blended = w * mat
        else:
            blended = blended + w * mat
        total_weight += w
        member_names.append(name)

    blended = blended / total_weight  # renormalize
    blended = blended / blended.sum()

    pred.members = member_names
    pred.source = "+".join(member_names)

    # ── probs come from the blended matrix ───────────────────────────────────
    pred.probs = outcome_probs(blended)

    # ── pool picks from blended matrix ───────────────────────────────────────
    p = picks.top_picks(blended, pts)
    pred.pool1, pred.pool2, pred.ep_table = p["pool1"], p["pool2"], p["table"]

    # ── expert picks display ──────────────────────────────────────────────────
    if raw_picks is not None:
        da, db = raw_picks["davo"]
        ma, mb = raw_picks["maldini"]
        pred.expert_picks = {
            "davo": f"{da}-{db}",
            "maldini": f"{ma}-{mb}",
        }

    # ── Elo-disagrees flag: compare Elo vs MARKET member 1X2 only ────────────
    if "market" in member_matrices and market_constraints:
        elo_probs = _elo_1x2_for(fixture, elo_ratings)
        if elo_probs:
            market_home_prob = market_constraints["1x2"]["home"]
            thr = cfg["elo"]["disagreement_threshold"]
            pred.elo_disagrees = abs(elo_probs["home"] - market_home_prob) > thr

    return pred


def predict_upcoming(fixtures_window, events, extras_by_event_id, elo_ratings,
                     cfg, odds_age=None, experts=None):
    preds = []
    for f in fixtures_window:
        e, swapped = match_event(f, events)
        extras = (extras_by_event_id or {}).get(e["id"]) if e else None
        preds.append(
            predict_match(f, e, extras, swapped, elo_ratings, cfg,
                          odds_age=odds_age, experts=experts)
        )
    return preds
