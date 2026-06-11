import numpy as np
import pytest

from model.picks import expected_points, outcome_of, top_picks

TOY = np.array([
    [0.20, 0.10, 0.05],
    [0.15, 0.20, 0.05],
    [0.10, 0.10, 0.05],
])  # rows = home goals; sums to 1.0

GROUP_PTS = {"exact": 5, "gd": 3, "winner": 2}


def test_outcome_of():
    assert outcome_of(2, 1) == "home"
    assert outcome_of(0, 0) == "draw"
    assert outcome_of(0, 3) == "away"


def test_expected_points_toy():
    rows = expected_points(TOY, GROUP_PTS)
    by_score = {r["score"]: r["ep"] for r in rows}
    # EP(0-0) = .20*5 + (.45-.20)*3 = 1.75 (draw: no winner tier)
    assert by_score["0-0"] == pytest.approx(1.75)
    assert by_score["1-1"] == pytest.approx(1.75)
    # EP(1-0) = .15*5 + (.25-.15)*3 + (.35-.25)*2 = 1.25
    assert by_score["1-0"] == pytest.approx(1.25)
    # EP(2-1) = .10*5 + (.25-.10)*3 + (.35-.25)*2 = 1.15
    assert by_score["2-1"] == pytest.approx(1.15)
    # EP(2-0) = .10*5 + (.10-.10)*3 + (.35-.10)*2 = 1.00
    assert by_score["2-0"] == pytest.approx(1.00)


def test_top_picks_tiebreak_and_pools():
    p = top_picks(TOY, GROUP_PTS, n=5)
    assert p["pool1"]["score"] == "0-0"   # EP tie with 1-1, fewer goals first
    assert p["pool2"]["score"] == "1-1"
    assert len(p["table"]) == 5
    assert p["table"][0] is p["pool1"]
