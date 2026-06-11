import pytest

from model.markets import build_constraints


def test_h2h_aggregated(sample_event):
    c = build_constraints(sample_event)
    # pinnacle devig home ~.586, bet365 ~.597 -> 50/50 blend ~.591
    assert c["1x2"]["home"] == pytest.approx(0.591, abs=0.01)
    assert sum(c["1x2"].values()) == pytest.approx(1.0)
    assert c["books_count"] == 2


def test_totals_collected(sample_event):
    c = build_constraints(sample_event)
    lines = dict(c["totals"])
    assert 2.5 in lines
    assert 0.3 < lines[2.5] < 0.6


def test_extras_add_btts_and_lines(sample_event, sample_extras):
    c = build_constraints(sample_event, sample_extras)
    assert c["btts"] == pytest.approx(0.392, abs=0.01)
    assert {ln for ln, _ in c["totals"]} == {1.5, 2.5, 3.5}


def test_integer_lines_skipped(sample_event):
    sample_event["bookmakers"][0]["markets"][1]["outcomes"] = [
        {"name": "Over", "point": 3, "price": 2.4},
        {"name": "Under", "point": 3, "price": 1.55},
    ]
    c = build_constraints(sample_event)
    lines = {ln for ln, _ in c["totals"]}
    assert 3 not in lines        # integer lines have push semantics -> excluded
    assert 2.5 in lines          # bet365's 2.5 line survives


def test_no_h2h_returns_none(sample_event):
    for b in sample_event["bookmakers"]:
        b["markets"] = [m for m in b["markets"] if m["key"] != "h2h"]
    assert build_constraints(sample_event) is None


def test_string_point_skipped(sample_event):
    sample_event["bookmakers"][0]["markets"][1]["outcomes"] = [
        {"name": "Over", "point": "2.5", "price": 2.10},
        {"name": "Under", "point": "2.5", "price": 1.78},
    ]
    c = build_constraints(sample_event)
    assert dict(c["totals"]).keys() == {2.5}  # only bet365's numeric line


def test_lowercase_draw_matched(sample_event):
    for b in sample_event["bookmakers"]:
        for m in b["markets"]:
            if m["key"] == "h2h":
                for o in m["outcomes"]:
                    if o["name"] == "Draw":
                        o["name"] = "draw"
    c = build_constraints(sample_event)
    assert c is not None and c["books_count"] == 2
