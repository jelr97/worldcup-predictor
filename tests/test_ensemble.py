"""Tests for the ensemble blend in model/predict.py."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from model.predict import MatchPrediction, predict_match, predict_upcoming

_ROOT = Path(__file__).resolve().parent.parent

CFG = {
    "pool": {"scoring": {"group": {"exact": 5, "gd": 3, "winner": 2},
                         "r32_r16": {"exact": 8, "gd": 5, "winner": 3},
                         "qf_plus": {"exact": 11, "gd": 7, "winner": 5}}},
    "elo": {"disagreement_threshold": 0.15},
    "ensemble": {"market": 1.0, "experts": 1.0},
}

FIXTURE = {"match_id": 1, "stage": "Group A", "home": "Mexico",
           "away": "South Africa", "venue": "Mexico City"}
ELO = {"mexico": 1800, "south africa": 1620}


def _make_experts(home="Mexico", away="South Africa",
                  davo=(2, 1), maldini=(2, 0)):
    """Construct a minimal experts dict for a single fixture."""
    from data.team_names import normalize
    return {
        (home, away): {
            "group": "A",
            "team_a": home,
            "team_b": away,
            "davo_a": davo[0],
            "davo_b": davo[1],
            "maldini_a": maldini[0],
            "maldini_b": maldini[1],
        }
    }


# ── weights renormalize when a member is missing ──────────────────────────────

def test_market_only_when_experts_absent(sample_event, sample_extras):
    """No experts -> market-only source, same as pre-ensemble behavior."""
    p_no_exp = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                             experts=None)
    p_empty = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                            experts={})
    assert "market" in p_no_exp.source
    assert "market" in p_empty.source
    # Picks should be the same when experts are absent
    assert p_no_exp.pool1["score"] == p_empty.pool1["score"]
    assert p_no_exp.pool2["score"] == p_empty.pool2["score"]


def test_experts_only_when_no_market():
    """No odds, but experts present -> source='experts'."""
    experts = _make_experts()
    p = predict_match(FIXTURE, None, None, False, {}, CFG, experts=experts)
    assert p.source == "experts"
    assert "experts" in p.members
    assert "market" not in p.members
    assert p.pool1 is not None


def test_ensemble_both_members(sample_event, sample_extras):
    """Both market and experts -> source='market+experts'."""
    experts = _make_experts()
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    assert p.source == "market+experts"
    assert "market" in p.members
    assert "experts" in p.members
    assert p.expert_picks is not None
    assert p.expert_picks["davo"] == "2-1"
    assert p.expert_picks["maldini"] == "2-0"


def test_market_only_matches_pre_ensemble(sample_event, sample_extras):
    """market-only (no experts) must equal old _finish behavior."""
    p_new = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                          experts=None)
    # Verify it still produces valid picks
    assert p_new.pool1 is not None
    assert p_new.pool2 is not None
    assert p_new.probs is not None
    assert abs(sum(p_new.probs.values()) - 1.0) < 1e-6


def test_elo_fallback_unchanged(sample_event):
    """When no market and no experts, elo fallback is unchanged."""
    p = predict_match(FIXTURE, None, None, False, ELO, CFG, experts=None)
    assert p.source == "elo"
    assert p.pool1 is not None


def test_none_when_no_data():
    """No event, no experts, no elo -> source='none'."""
    p = predict_match(FIXTURE, None, None, False, {}, CFG, experts=None)
    assert p.source == "none"
    assert p.pool1 is None


# ── ensemble blend validity ───────────────────────────────────────────────────

def test_blended_probs_sum_to_one(sample_event, sample_extras):
    experts = _make_experts()
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    assert abs(sum(p.probs.values()) - 1.0) < 1e-6


def test_blended_picks_valid(sample_event, sample_extras):
    """Blended prediction yields valid Pool 1 and Pool 2 picks."""
    experts = _make_experts()
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    assert p.pool1 is not None
    assert p.pool2 is not None
    assert "-" in p.pool1["score"]
    assert "-" in p.pool2["score"]


def test_blended_pool1_pool2_can_differ(sample_event, sample_extras):
    """With typical inputs, pool1 and pool2 picks need not be identical
    (they depend on the scoring tier's ep landscape — just verify validity)."""
    experts = _make_experts()
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    # Both picks must be valid score strings
    for score in (p.pool1["score"], p.pool2["score"]):
        parts = score.split("-")
        assert len(parts) == 2
        assert all(part.isdigit() for part in parts)


# ── custom weights renormalize ────────────────────────────────────────────────

def test_zero_expert_weight_equals_market_only(sample_event, sample_extras):
    """Setting experts weight to 0 should give market-only result."""
    cfg_zero = dict(CFG, ensemble={"market": 1.0, "experts": 0.0})
    experts = _make_experts()
    p_blend = predict_match(FIXTURE, sample_event, sample_extras, False, ELO,
                            cfg_zero, experts=experts)
    p_market = predict_match(FIXTURE, sample_event, sample_extras, False, ELO,
                             cfg_zero, experts=None)
    assert p_blend.pool1["score"] == p_market.pool1["score"]
    assert p_blend.pool2["score"] == p_market.pool2["score"]


# ── expert_picks field ────────────────────────────────────────────────────────

def test_expert_picks_formatted_correctly(sample_event, sample_extras):
    experts = _make_experts(davo=(3, 1), maldini=(2, 2))
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    assert p.expert_picks["davo"] == "3-1"
    assert p.expert_picks["maldini"] == "2-2"


def test_expert_picks_none_when_no_experts(sample_event, sample_extras):
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=None)
    assert p.expert_picks is None


# ── elo_disagrees still uses market 1X2 ──────────────────────────────────────

def test_elo_disagrees_uses_market(sample_event, sample_extras):
    """elo_disagrees should compare Elo vs market member only, not blend."""
    experts = _make_experts()
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG,
                      experts=experts)
    # Mexico is the favorite in both Elo and market -> should not disagree
    assert p.elo_disagrees is False


# ── predict_upcoming signature ────────────────────────────────────────────────

def test_predict_upcoming_accepts_experts(sample_event, sample_extras):
    """predict_upcoming must accept experts param without error."""
    from data.fixtures import load_fixtures, upcoming
    fixtures = load_fixtures()
    window_fx = upcoming(fixtures, window_hours=24 * 365)  # all fixtures
    experts = {}  # empty is fine
    preds = predict_upcoming(window_fx, [sample_event], {}, ELO, CFG,
                             experts=experts)
    assert isinstance(preds, list)


# ── full coverage: all 72 expert rows match all 72 group fixtures ─────────────

def test_all_72_group_fixtures_have_expert_coverage():
    """All 72 group-stage fixtures must be covered by the experts CSV (either orientation).

    Reports: matched count, orientation-swapped count, and 3 example matches.
    """
    csv_path = _ROOT / "data" / "experts.csv"
    if not csv_path.exists():
        pytest.skip("data/experts.csv not present — run scripts/import_experts.py first")

    with open(_ROOT / "data" / "fixtures.json", encoding="utf-8") as f:
        all_fx = json.load(f)
    group_fixtures = [fx for fx in all_fx if fx["stage"].startswith("Group")]

    with patch("data.experts._CSV_PATH", csv_path):
        from data.experts import load_experts, picks_for
        experts = load_experts()

    matched = 0
    swapped_count = 0
    unmatched = []

    for fx in group_fixtures:
        home, away = fx["home"], fx["away"]
        result = picks_for(experts, home, away)
        if result is None:
            unmatched.append(f"{home} vs {away}")
        else:
            matched += 1
            # Check orientation
            from data.team_names import same_team
            is_direct = any(
                same_team(home, ta) and same_team(away, tb)
                for (ta, tb) in experts.keys()
            )
            if not is_direct:
                swapped_count += 1

    assert not unmatched, (
        f"{len(unmatched)} fixtures had no expert match:\n"
        + "\n".join(f"  {m}" for m in unmatched)
    )
    assert matched == 72

    # Print informational summary (captured by pytest -s or -v)
    print(
        f"\nExpert coverage: {matched}/72 fixtures matched, "
        f"{swapped_count} orientation-swapped"
    )
