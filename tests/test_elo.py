import pytest

from data.elo import elo_1x2, load_ratings, rating_for, win_expectancy


def test_load_ratings(tmp_path):
    p = tmp_path / "elo.csv"
    p.write_text("team,rating\nBrazil,2100\nUSA,1790\n", encoding="utf-8")
    r = load_ratings(p)
    assert r["brazil"] == 2100
    assert rating_for(r, "United States") == 1790  # via alias


def test_missing_snapshot_is_empty(tmp_path):
    assert load_ratings(tmp_path / "nope.csv") == {}


def test_win_expectancy():
    assert win_expectancy(1800, 1800) == pytest.approx(0.5)
    assert win_expectancy(2000, 1600) > 0.9


def test_elo_1x2_balanced():
    p = elo_1x2(0.5)
    assert p["draw"] == pytest.approx(0.27, abs=0.01)
    assert p["home"] == pytest.approx(p["away"])
    assert sum(p.values()) == pytest.approx(1.0)


def test_elo_1x2_favorite():
    p = elo_1x2(0.9)
    assert p["home"] > 0.8
    assert sum(p.values()) == pytest.approx(1.0)
