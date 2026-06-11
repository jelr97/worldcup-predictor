import numpy as np
import pytest

from model.picks import expected_points, outcome_of, top_picks

TOY = np.array([
    [0.20, 0.10, 0.05],
    [0.15, 0.20, 0.05],
    [0.10, 0.10, 0.05],
])  # rows = home goals; sums to 1.0


def test_outcome_of():
    assert outcome_of(2, 1) == "home"
    assert outcome_of(0, 0) == "draw"
    assert outcome_of(0, 3) == "away"


def test_expected_points_toy():
    rows = expected_points(TOY, pts_exact=3, pts_outcome=1)
    by_score = {r["score"]: r["ep"] for r in rows}
    assert by_score["0-0"] == pytest.approx(0.85)
    assert by_score["1-1"] == pytest.approx(0.85)
    assert by_score["1-0"] == pytest.approx(0.65)


def test_top_picks_tiebreak_and_pools():
    p = top_picks(TOY, pts_exact=3, pts_outcome=1, n=5)
    assert p["pool1"]["score"] == "0-0"   # EP tie with 1-1, fewer goals first
    assert p["pool2"]["score"] == "1-1"
    assert len(p["table"]) == 5
    assert p["table"][0] is p["pool1"]
