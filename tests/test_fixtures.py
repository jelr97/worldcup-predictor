import json
from datetime import datetime, timezone

from data.fixtures import load_fixtures, upcoming

SAMPLE = [
    {"match_id": 2, "stage": "Group B", "home": "Canada", "away": "Qatar",
     "kickoff_utc": "2026-06-12T01:00:00Z", "venue": "Toronto"},
    {"match_id": 1, "stage": "Group A", "home": "Mexico", "away": "South Africa",
     "kickoff_utc": "2026-06-11T19:00:00Z", "venue": "Mexico City"},
]


def _write(tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return p


def test_load_sorts_and_converts_tz(tmp_path):
    fx = load_fixtures(_write(tmp_path))
    assert [f["match_id"] for f in fx] == [1, 2]
    # 19:00 UTC on Jun 11 is 15:00 EDT
    assert fx[0]["kickoff_et"].hour == 15


def test_upcoming_window(tmp_path):
    fx = load_fixtures(_write(tmp_path))
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    assert len(upcoming(fx, now=now, window_hours=12)) == 1
    assert len(upcoming(fx, now=now, window_hours=48)) == 2
    assert len(upcoming(fx, now=datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc),
                        window_hours=48)) == 0
