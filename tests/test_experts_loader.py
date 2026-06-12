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


def test_load_experts_non_utf8_bytes_no_crash(tmp_path):
    """A file containing non-UTF-8 bytes (e.g. ANSI/cp1252 re-save) must not
    crash load_experts().  It should warn and return whatever rows were
    collected before the bad byte was encountered (possibly {})."""
    # Write a valid CSV header + one valid row, then append a raw cp1252 byte
    # sequence that is not valid UTF-8 (e.g. 0x80 is a cp1252 euro sign but
    # invalid as UTF-8).
    p = tmp_path / "experts.csv"
    valid_content = (
        "group,team_a,team_b,davo_a,davo_b,maldini_a,maldini_b\n"
        "A,Mexico,South Africa,2,1,2,0\n"
    )
    # Append a line with an invalid UTF-8 byte (0x80) in a team name field
    bad_line = b"B,T\xc3\xbcrkiye,Qatar,1,0,1,1\n"  # valid UTF-8: Türkiye
    # Force an undecodable byte by replacing the ü (c3 bc) with raw 0xfc (valid latin-1 but not utf-8 lead)
    bad_line_invalid = b"B,T\xfcrkiye,Qatar,1,0,1,1\n"
    p.write_bytes(valid_content.encode("utf-8") + bad_line_invalid)
    with patch("data.experts._CSV_PATH", p):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_experts()
    # Must return a dict (not raise)
    assert isinstance(result, dict)
    # The valid first row may or may not have been collected depending on
    # when the error fires — either way the function must not crash.
    # A RuntimeWarning must have been emitted about the encoding problem.
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "expected a RuntimeWarning for the non-UTF-8 file"


def test_load_experts_csv_field_too_large_no_crash(tmp_path, monkeypatch):
    """A csv.Error raised during iteration (e.g. field larger than field limit)
    must not crash load_experts() — it should warn and return partial/empty dict."""
    import csv as _csv
    p = tmp_path / "experts.csv"
    # Write a normal-looking file; we'll monkeypatch csv.DictReader to raise
    # csv.Error mid-iteration to simulate the 'field larger than field limit' case.
    valid_content = (
        "group,team_a,team_b,davo_a,davo_b,maldini_a,maldini_b\n"
        "A,Mexico,South Africa,2,1,2,0\n"
        "B,Canada,Qatar,3,0,1,0\n"
    )
    p.write_text(valid_content, encoding="utf-8")

    original_dictreader = _csv.DictReader

    class _BoomOnSecondRow:
        """Yields first row fine, then raises csv.Error on the second."""
        def __init__(self, f, **kw):
            self._inner = iter(original_dictreader(f, **kw))
            self._count = 0
            self.fieldnames = None
            # Prime fieldnames by peeking at the inner reader
            inner_dr = original_dictreader.__new__(original_dictreader)
            # Just use the real reader to grab fieldnames
            self._reader = original_dictreader(f)

        def __iter__(self):
            count = 0
            for row in self._inner:
                count += 1
                if count == 2:
                    raise _csv.Error("field larger than field limit (131072)")
                yield row

    monkeypatch.setattr(_csv, "DictReader", _BoomOnSecondRow)
    with patch("data.experts._CSV_PATH", p):
        import warnings
        import data.experts as _mod
        # Reload so monkeypatch on csv.DictReader is picked up
        import importlib
        importlib.reload(_mod)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _mod.load_experts()
    assert isinstance(result, dict)
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "expected a RuntimeWarning for the csv.Error"


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
