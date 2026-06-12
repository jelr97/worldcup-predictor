"""Tests for model.scoring.score_pick — hand-computed across all three tiers
and two stages (group and qf_plus)."""
import pytest

from model.scoring import score_pick

# ── group stage points ────────────────────────────────────────────────────────
GROUP = {"exact": 5, "gd": 3, "winner": 2}
# ── knockout QF+ stage points ─────────────────────────────────────────────────
QF = {"exact": 11, "gd": 7, "winner": 5}


# ── exact score tier ──────────────────────────────────────────────────────────

def test_exact_score_group():
    assert score_pick("2-1", "2-1", GROUP) == 5


def test_exact_score_qf():
    assert score_pick("0-0", "0-0", QF) == 11


def test_exact_score_home_win():
    assert score_pick("3-0", "3-0", GROUP) == 5


def test_exact_score_away_win():
    assert score_pick("0-2", "0-2", QF) == 11


# ── goal-difference tier ──────────────────────────────────────────────────────

def test_gd_tier_group():
    # Pick 2-1 (GD=+1); actual 3-2 (GD=+1) — same GD, not exact
    assert score_pick("2-1", "3-2", GROUP) == 3


def test_gd_tier_qf():
    assert score_pick("1-0", "2-1", QF) == 7


def test_gd_draw_predicted_draw_occurred():
    # Pick 1-1 (GD=0); actual 0-0 (GD=0)
    assert score_pick("1-1", "0-0", GROUP) == 3


def test_gd_draw_predicted_matches_any_draw():
    # Pick 0-0; actual 2-2 — both draws (GD=0 == GD=0)
    assert score_pick("0-0", "2-2", QF) == 7


# ── winner-only tier ─────────────────────────────────────────────────────────

def test_winner_tier_group():
    # Pick 1-0 (home win, GD=+1); actual 3-1 (home win, GD=+2) — right winner, wrong GD
    assert score_pick("1-0", "3-1", GROUP) == 2


def test_winner_tier_qf():
    # Pick 1-0 (home win, GD=+1); actual 3-1 (home win, GD=+2) — right winner only
    assert score_pick("1-0", "3-1", QF) == 5


def test_winner_tier_away():
    # Pick 0-2 (away win); actual 0-1 (away win, different GD)
    assert score_pick("0-2", "0-1", GROUP) == 2


# ── zero points (wrong winner) ────────────────────────────────────────────────

def test_wrong_winner_zero():
    # Pick home win; actual away win
    assert score_pick("2-0", "0-1", GROUP) == 0


def test_draw_pick_nondraw_result_zero():
    # Pick 1-1 (draw); actual 2-0 (home win)
    assert score_pick("1-1", "2-0", GROUP) == 0


def test_home_pick_draw_result_zero():
    # Pick 2-0 (home win); actual 1-1 (draw)
    assert score_pick("2-0", "1-1", QF) == 0


# ── stage independence ────────────────────────────────────────────────────────

def test_same_pick_different_stages():
    r32 = {"exact": 8, "gd": 5, "winner": 3}
    # Exact score gives different points by stage
    assert score_pick("1-0", "1-0", GROUP) == 5
    assert score_pick("1-0", "1-0", r32) == 8
    assert score_pick("1-0", "1-0", QF) == 11
