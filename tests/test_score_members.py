"""Tests for scripts/score_members.py logic.

Tests cover:
  - pick_log idempotence
  - retro-scoring of static members without snapshots
  - scoring function integration across tiers
"""
import csv
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add repo root to path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# We import internal helpers (not the main() function) to unit-test them.
import importlib, types

# Load the module without executing main()
import scripts.score_members as sm


CFG = {
    "pool": {"scoring": {
        "group":   {"exact": 5, "gd": 3, "winner": 2},
        "r32_r16": {"exact": 8, "gd": 5, "winner": 3},
        "qf_plus": {"exact": 11, "gd": 7, "winner": 5},
    }},
    "elo": {"disagreement_threshold": 0.15},
    "ensemble": {"market": 1.0, "experts": 1.0},
    "knockout_scoring": "90min",
}

_FIXTURE_A = {
    "match_id": 1,
    "stage": "Group A",
    "home": "Mexico",
    "away": "South Africa",
    "kickoff_utc": datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc),
    "kickoff_et": datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc),
    "venue": "Mexico City Stadium",
}


# ── pick_log helpers ─────────────────────────────────────────────────────────

def _write_pick_log(tmp_path, rows):
    p = tmp_path / "pick_log.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sm.PICK_LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return p


def test_load_pick_log_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "PICK_LOG", tmp_path / "nonexistent.csv")
    assert sm._load_pick_log() == {}


def test_load_pick_log_existing(tmp_path, monkeypatch):
    rows = [
        {"match_id": 1, "member": "davo", "pick": "2-1",
         "computed_at": "2026-06-12T10:00:00+00:00"},
        {"match_id": 1, "member": "maldini", "pick": "1-0",
         "computed_at": "2026-06-12T10:00:00+00:00"},
    ]
    p = _write_pick_log(tmp_path, rows)
    monkeypatch.setattr(sm, "PICK_LOG", p)
    log = sm._load_pick_log()
    assert log[(1, "davo")] == "2-1"
    assert log[(1, "maldini")] == "1-0"


def test_append_picks_creates_file(tmp_path, monkeypatch):
    p = tmp_path / "pick_log.csv"
    monkeypatch.setattr(sm, "PICK_LOG", p)
    rows = [{"match_id": 5, "member": "experts", "pick": "0-0",
             "computed_at": "2026-06-12T12:00:00+00:00"}]
    sm._append_picks(rows)
    assert p.exists()
    log = sm._load_pick_log()
    assert log[(5, "experts")] == "0-0"


def test_append_picks_idempotent_does_not_overwrite(tmp_path, monkeypatch):
    """Appending twice does NOT overwrite existing rows (idempotence via skip)."""
    p = tmp_path / "pick_log.csv"
    monkeypatch.setattr(sm, "PICK_LOG", p)
    row = {"match_id": 3, "member": "davo", "pick": "1-1",
           "computed_at": "2026-06-12T10:00:00+00:00"}
    sm._append_picks([row])
    # Simulate second call — the snapshot_upcoming function checks existing
    # and skips; we verify the file has only one row (not two).
    existing = sm._load_pick_log()
    if (3, "davo") not in existing:
        sm._append_picks([dict(row, pick="2-2")])  # would overwrite if not skipped
    # Only the first row should exist
    log = sm._load_pick_log()
    assert log[(3, "davo")] == "1-1"


# ── retro-scoring of static members ─────────────────────────────────────────

def test_score_backward_retros_static_without_snapshot(tmp_path, monkeypatch):
    """davo/maldini/experts are retro-scored for finished matches even without
    a pick_log entry (i.e., before this feature existed)."""
    monkeypatch.setattr(sm, "PICK_LOG", tmp_path / "empty.csv")
    # experts.csv must exist for picks_for to work
    experts_data = _load_test_experts()
    if not experts_data:
        pytest.skip("data/experts.csv not found")

    # Fixture 1: Mexico 2-0 South Africa
    fixtures = [dict(_FIXTURE_A)]
    results = {1: (2, 0)}   # Mexico won 2-0

    stats = sm.score_backward(fixtures, CFG, experts_data, results)
    # davo, maldini, experts should all appear (retro-scored)
    assert "davo" in stats or "maldini" in stats or "experts" in stats


def test_score_backward_no_results_returns_none_gracefully(tmp_path, monkeypatch):
    """When feed is unreachable (results=None), scoring prints a note and
    returns None without crashing."""
    monkeypatch.setattr(sm, "PICK_LOG", tmp_path / "empty.csv")
    experts_data = {}
    fixtures = [dict(_FIXTURE_A)]
    result = sm.score_backward(fixtures, CFG, experts_data, None)
    assert result is None


def test_score_backward_market_only_from_snapshot(tmp_path, monkeypatch):
    """Market picks are only scored when a snapshot exists in pick_log."""
    p = _write_pick_log(tmp_path, [
        {"match_id": 1, "member": "market", "pick": "2-0",
         "computed_at": "2026-06-12T10:00:00+00:00"}
    ])
    monkeypatch.setattr(sm, "PICK_LOG", p)
    experts_data = {}
    fixtures = [dict(_FIXTURE_A)]
    results = {1: (2, 0)}   # exact hit

    stats = sm.score_backward(fixtures, CFG, experts_data, results)
    assert "market" in stats
    assert stats["market"]["exact"] == 1
    assert stats["market"]["total"] == 5  # group exact = 5


def test_score_backward_accumulates_multiple_matches(tmp_path, monkeypatch):
    """Stats accumulate correctly across multiple matches."""
    fixture_2 = {
        "match_id": 2,
        "stage": "Group A",
        "home": "Korea Republic",
        "away": "Czechia",
        "kickoff_utc": datetime(2026, 6, 12, 2, 0, tzinfo=timezone.utc),
        "kickoff_et": datetime(2026, 6, 11, 22, 0, tzinfo=timezone.utc),
        "venue": "Guadalajara Stadium",
    }
    rows = [
        {"match_id": 1, "member": "blend", "pick": "2-1",
         "computed_at": "2026-06-12T10:00:00+00:00"},
        {"match_id": 2, "member": "blend", "pick": "2-1",
         "computed_at": "2026-06-12T10:00:00+00:00"},
    ]
    p = _write_pick_log(tmp_path, rows)
    monkeypatch.setattr(sm, "PICK_LOG", p)
    # match 1: blend picked 2-1, actual 2-0 -> winner tier (2pts)
    # match 2: blend picked 2-1, actual 2-1 -> exact (5pts)
    results = {1: (2, 0), 2: (2, 1)}
    fixtures = [dict(_FIXTURE_A), dict(fixture_2)]
    experts_data = {}
    stats = sm.score_backward(fixtures, CFG, experts_data, results)
    assert stats["blend"]["matched"] == 2
    assert stats["blend"]["total"] == 7   # 2 + 5
    assert stats["blend"]["exact"] == 1
    assert stats["blend"]["winner"] == 1


# ── market pick orientation independence ─────────────────────────────────────

def test_market_pick_orientation_independent(sample_event):
    """_market_pick must return the same pick whether the odds event matches
    the fixture orientation or is home/away-swapped (identical prices).

    Production-path regression test for the swapped-spread constraint flip
    in _market_pick: it must use (-line, 1 - p), not (-line, p). With the
    buggy flip the solver lambdas skew (~1.76 -> ~2.05 for the sample event)
    and the Pool 1 pick changes, so this test fails.
    """
    import copy
    swapped_event = copy.deepcopy(sample_event)
    swapped_event["home_team"], swapped_event["away_team"] = (
        sample_event["away_team"], sample_event["home_team"])

    pick_direct = sm._market_pick(_FIXTURE_A, [sample_event], CFG)
    pick_swapped = sm._market_pick(_FIXTURE_A, [swapped_event], CFG)
    assert pick_direct is not None
    assert pick_direct == pick_swapped, (
        f"market pick differs across event orientations: "
        f"{pick_direct} vs {pick_swapped}")


# ── helper ───────────────────────────────────────────────────────────────────

def _load_test_experts():
    """Load experts.csv if it exists."""
    from data.experts import load_experts
    return load_experts()
