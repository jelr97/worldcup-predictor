import pytest

from model.implied import aggregate, devig


def test_devig_sums_to_one():
    p = devig({"home": 1.65, "draw": 3.9, "away": 5.8})
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home"] == pytest.approx(0.586, abs=0.005)


def test_devig_rejects_bad_odds():
    assert devig({}) is None
    assert devig({"home": 1.0, "away": 2.0}) is None
    assert devig({"home": None, "away": 2.0}) is None


def test_aggregate_median():
    books = {
        "a": {"home": 0.50, "draw": 0.30, "away": 0.20},
        "b": {"home": 0.60, "draw": 0.25, "away": 0.15},
        "c": {"home": 0.55, "draw": 0.28, "away": 0.17},
    }
    agg = aggregate(books)
    assert agg["home"] == pytest.approx(0.55, abs=0.01)
    assert sum(agg.values()) == pytest.approx(1.0)


def test_aggregate_pinnacle_weighted():
    books = {
        "pinnacle": {"home": 0.60, "draw": 0.25, "away": 0.15},
        "b": {"home": 0.50, "draw": 0.30, "away": 0.20},
    }
    agg = aggregate(books)
    # 0.5*pinnacle + 0.5*median(others) = 0.5*0.60 + 0.5*0.50 = 0.55
    assert agg["home"] == pytest.approx(0.55, abs=0.01)


def test_aggregate_empty():
    assert aggregate({}) is None
    assert aggregate({"a": None}) is None


def test_aggregate_mismatched_keys_uses_shared():
    books = {
        "a": {"home": 0.50, "draw": 0.30, "away": 0.20},
        "b": {"home": 0.60, "away": 0.40},
    }
    agg = aggregate(books)
    assert set(agg) == {"home", "away"}
    assert sum(agg.values()) == pytest.approx(1.0)


def test_aggregate_pinnacle_only():
    agg = aggregate({"pinnacle": {"home": 0.6, "draw": 0.25, "away": 0.15}})
    assert agg["home"] == pytest.approx(0.6)
