"""Tests for expert-related UI rendering in ui.py."""
import html as html_mod
from datetime import datetime, timezone

import pytest

from model.predict import MatchPrediction
from ui import render_card


def _make_fixture(home="Mexico", away="South Africa"):
    return {
        "home": home,
        "away": away,
        "stage": "Group A",
        "kickoff_et": datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc),
        "venue": "Test Stadium",
    }


def _make_pred(**kwargs):
    defaults = dict(
        fixture=_make_fixture(),
        source="market",
        probs={"home": 0.67, "draw": 0.22, "away": 0.11},
        pool1={"score": "1-0"},
        pool2={"score": "2-0"},
        ep_table=[],
        books_count=42,
        elo_disagrees=False,
        note="",
        members=["market"],
        expert_picks=None,
    )
    defaults.update(kwargs)
    return MatchPrediction(**defaults)


# ── expert_picks shown in caption ─────────────────────────────────────────────

def test_expert_picks_in_caption():
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "2-1", "maldini": "2-0"},
    )
    out = render_card(p, "🇲🇽", "🇿🇦")
    assert "Davo 2-1" in out
    assert "Maldini 2-0" in out


def test_expert_picks_absent_when_none():
    p = _make_pred(source="market", members=["market"], expert_picks=None)
    out = render_card(p, "🇲🇽", "🇿🇦")
    assert "Davo" not in out
    assert "Maldini" not in out


def test_expert_picks_xss_escaped():
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "<script>", "maldini": "<b>0-0</b>"},
    )
    out = render_card(p, "", "")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<b>" not in out


# ── bookmakers shown only when market is a member ─────────────────────────────

def test_bookmakers_shown_with_market_member():
    p = _make_pred(source="market+experts", members=["market", "experts"],
                   books_count=15)
    out = render_card(p, "", "")
    assert "15 bookmakers" in out


def test_bookmakers_not_shown_experts_only():
    p = _make_pred(source="experts", members=["experts"], books_count=0,
                   probs={"home": 0.60, "draw": 0.25, "away": 0.15})
    out = render_card(p, "", "")
    assert "bookmakers" not in out


def test_bookmakers_shown_market_only():
    p = _make_pred(source="market", members=["market"], books_count=7)
    out = render_card(p, "", "")
    assert "7 bookmakers" in out


# ── elo source: no expert caption ────────────────────────────────────────────

def test_elo_source_no_expert_caption():
    p = _make_pred(source="elo", members=["elo"],
                   expert_picks={"davo": "1-0", "maldini": "1-0"})
    out = render_card(p, "", "")
    # For elo source the experts section shouldn't render
    assert "model-only" in out


# ── backward compat: empty members list ──────────────────────────────────────

def test_backward_compat_empty_members():
    """MatchPrediction with source='market' and empty members list shows bookmakers."""
    p = _make_pred(source="market", members=[], books_count=30)
    out = render_card(p, "", "")
    assert "30 bookmakers" in out
