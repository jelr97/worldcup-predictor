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


# ── expert_picks shown as chips under badges ──────────────────────────────────

def test_expert_chips_rendered():
    """Expert picks appear as DAVO / MALDINI chips after the POOL badges."""
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "2-1", "maldini": "2-0"},
    )
    out = render_card(p, "🇲🇽", "🇿🇦")
    assert "DAVO 2-1" in out
    assert "MALDINI 2-0" in out


def test_expert_chips_absent_when_none():
    """No chips rendered when expert_picks is None."""
    p = _make_pred(source="market", members=["market"], expert_picks=None)
    out = render_card(p, "🇲🇽", "🇿🇦")
    assert "DAVO" not in out
    assert "MALDINI" not in out


def test_expert_chips_order_after_badges():
    """Chip row comes after the POOL badge row in the HTML."""
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "2-1", "maldini": "2-0"},
    )
    out = render_card(p, "🇲🇽", "🇿🇦")
    pool_pos = out.index("POOL 1")
    davo_pos = out.index("DAVO 2-1")
    assert davo_pos > pool_pos


def test_expert_chips_xss_escaped():
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "<script>", "maldini": "<b>0-0</b>"},
    )
    out = render_card(p, "", "")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<b>" not in out


def test_expert_picks_not_in_caption():
    """Expert picks must NOT appear in the caption line (moved to chip row)."""
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        expert_picks={"davo": "2-1", "maldini": "2-0"},
    )
    out = render_card(p, "🇲🇽", "🇿🇦")
    # The caption div ends before the badges div; DAVO chips come after badges
    # so they cannot be in the caption. We verify the caption segment:
    # caption is the div with font-size:12px; chips use 13px and border.
    assert "Davo 2-1" not in out   # old caption text style (lowercase Davo)
    assert "Maldini 2-0" not in out  # old caption text style


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


# ── bookmaker-only picks chip ─────────────────────────────────────────────────

def test_bookies_row_shown_when_blended():
    """Market+experts blend shows a BOOKMAKERS ONLY row with market-only picks."""
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        market_pool1={"score": "1-0"},
        market_pool2={"score": "2-1"},
    )
    out = render_card(p, "", "")
    assert "BOOKMAKERS ONLY" in out
    assert "1-0" in out and "2-1" in out


def test_bookies_row_shown_when_market_only():
    """Market-only: still labelled explicitly so the bookmaker pick is findable."""
    p = _make_pred(
        source="market",
        members=["market"],
        market_pool1={"score": "1-0"},
        market_pool2={"score": "2-0"},
    )
    out = render_card(p, "", "")
    assert "BOOKMAKERS ONLY" in out


def test_bookies_row_absent_when_experts_only():
    """Experts-only: no market picks, so no BOOKMAKERS ONLY row."""
    p = _make_pred(
        source="experts",
        members=["experts"],
        probs={"home": 0.60, "draw": 0.25, "away": 0.15},
        market_pool1=None,
        market_pool2=None,
    )
    out = render_card(p, "", "")
    assert "BOOKMAKERS ONLY" not in out


def test_bookies_row_xss_escaped():
    p = _make_pred(
        source="market+experts",
        members=["market", "experts"],
        market_pool1={"score": "<script>"},
        market_pool2={"score": "<b>0-0</b>"},
    )
    out = render_card(p, "", "")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


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
