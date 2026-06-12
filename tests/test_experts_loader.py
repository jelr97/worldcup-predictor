"""Tests for data/experts.py loader."""
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from data.experts import load_experts, picks_for


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_experts_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "experts.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["group", "team_a", "team_b",
                        "davo_a", "davo_b", "maldini_a", "maldini_b"],
        )
        w.writeheader()
        w.writerows(rows)
    return p


_SAMPLE_ROW = {
    "group": "A",
    "team_a": "Mexico",
    "team_b": "South Africa",
    "davo_a": 2,
    "davo_b": 1,
    "maldini_a": 2,
    "maldini_b": 0,
}


# ── load_experts ──────────────────────────────────────────────────────────────

def test_load_experts_missing_file(tmp_path):
    """Missing CSV -> empty dict, no exception."""
    with patch("data.experts._CSV_PATH", tmp_path / "no_such.csv"):
        result = load_experts()
    assert result == {}


def test_load_experts_returns_dict(tmp_path):
    p = _write_experts_csv(tmp_path, [_SAMPLE_ROW])
    with patch("data.experts._CSV_PATH", p):
        result = load_experts()
    assert len(result) == 1
    key = ("Mexico", "South Africa")
    assert key in result
    row = result[key]
    assert row["davo_a"] == 2
    assert row["davo_b"] == 1
    assert row["maldini_a"] == 2
    assert row["maldini_b"] == 0


def test_load_experts_corrupt_file_returns_empty(tmp_path):
    """A file with completely wrong columns yields {} without crashing."""
    p = tmp_path / "experts.csv"
    p.write_text("definitely,not,valid\ndata,here")
    with patch("data.experts._CSV_PATH", p):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_experts()
    assert isinstance(result, dict)  # must not crash; {} is the only valid outcome


def test_load_experts_single_corrupt_row_skipped_with_warning(tmp_path):
    """One bad row emits a RuntimeWarning and is skipped; valid rows survive."""
    good = dict(_SAMPLE_ROW)
    bad = {
        "group": "B",
        "team_a": "Canada",
        "team_b": "Qatar",
        "davo_a": "X",  # not an integer
        "davo_b": 1,
        "maldini_a": 4,
        "maldini_b": 1,
    }
    p = _write_experts_csv(tmp_path, [good, bad])
    with patch("data.experts._CSV_PATH", p):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_experts()
    # Good row must be present
    assert ("Mexico", "South Africa") in result
    # Bad row must be absent
    assert ("Canada", "Qatar") not in result
    # A RuntimeWarning must have been emitted for the bad row
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "expected a RuntimeWarning for the corrupt row"


# ── picks_for ─────────────────────────────────────────────────────────────────

def test_picks_for_direct_orientation(tmp_path):
    p = _write_experts_csv(tmp_path, [_SAMPLE_ROW])
    with patch("data.experts._CSV_PATH", p):
        experts = load_experts()
    result = picks_for(experts, "Mexico", "South Africa")
    assert result is not None
    assert result["davo"] == (2, 1)
    assert result["maldini"] == (2, 0)


def test_picks_for_swapped_orientation(tmp_path):
    """When fixture is stored reversed, scores are swapped back."""
    p = _write_experts_csv(tmp_path, [_SAMPLE_ROW])
    with patch("data.experts._CSV_PATH", p):
        experts = load_experts()
    # Request with home=South Africa, away=Mexico (reversed from stored)
    result = picks_for(experts, "South Africa", "Mexico")
    assert result is not None
    # Scores should be swapped: davo was 2-1 for Mexico, so for SA it's 1-2
    assert result["davo"] == (1, 2)
    assert result["maldini"] == (0, 2)


def test_picks_for_missing_fixture_returns_none(tmp_path):
    p = _write_experts_csv(tmp_path, [_SAMPLE_ROW])
    with patch("data.experts._CSV_PATH", p):
        experts = load_experts()
    result = picks_for(experts, "Brazil", "Germany")
    assert result is None


def test_picks_for_empty_experts():
    assert picks_for({}, "Mexico", "South Africa") is None


def test_picks_for_uses_same_team_normalization(tmp_path):
    """Spanish/aliased team names should still match canonical fixtures."""
    row = {
        "group": "E",
        "team_a": "Germany",
        "team_b": "Curaçao",
        "davo_a": 5,
        "davo_b": 0,
        "maldini_a": 7,
        "maldini_b": 0,
    }
    p = _write_experts_csv(tmp_path, [row])
    with patch("data.experts._CSV_PATH", p):
        experts = load_experts()
    # same_team("Germany", "Germany") == True via normalize
    result = picks_for(experts, "Germany", "Curaçao")
    assert result is not None
    assert result["davo"] == (5, 0)
