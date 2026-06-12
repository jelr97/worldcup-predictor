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


# ── spreads parsing ───────────────────────────────────────────────────────────

def test_spreads_parsed_from_event(sample_event):
    """Sample event has home -1.5 from both books; constraint should appear."""
    c = build_constraints(sample_event)
    assert "spreads" in c
    lines = dict(c["spreads"])
    assert -1.5 in lines
    p = lines[-1.5]
    # devigged home-covers probability: pinnacle 2.55/1.52 -> home ~0.373;
    # bet365 2.60/1.50 -> home ~0.366; 50/50 blend with Pinnacle weighting ~0.37
    assert 0.30 < p < 0.45


def test_spreads_integer_line_skipped(sample_event):
    """Integer spread lines must be skipped (push semantics)."""
    # Replace the spreads market in pinnacle ONLY with an integer line (-2)
    for b in sample_event["bookmakers"]:
        if b["key"] == "pinnacle":
            for m in b["markets"]:
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        o["point"] = -2 if o["name"] == "Mexico" else 2
    c = build_constraints(sample_event)
    # pinnacle's -2 is an integer line -> skipped; bet365 still has -1.5
    lines = dict(c["spreads"])
    assert -2 not in lines
    assert -1.5 in lines


def test_spreads_absent_when_no_spreads_market(sample_event):
    """If no book quotes spreads, constraint list is empty (not absent)."""
    for b in sample_event["bookmakers"]:
        b["markets"] = [m for m in b["markets"] if m["key"] != "spreads"]
    c = build_constraints(sample_event)
    assert c["spreads"] == []


def test_spreads_swapped_orientation():
    """When odds event is home/away-swapped vs fixture, lines flip sign."""
    import copy
    # Build a raw event where South Africa is listed as home_team
    event = {
        "id": "swap_test",
        "home_team": "South Africa",
        "away_team": "Mexico",
        "bookmakers": [
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "South Africa", "price": 5.8},
                    {"name": "Draw", "price": 3.9},
                    {"name": "Mexico", "price": 1.65}]},
                {"key": "spreads", "outcomes": [
                    # South Africa is event-home with +1.5 (event-home handicap = +1.5)
                    {"name": "South Africa", "point": 1.5, "price": 1.52},
                    # Mexico is event-away with -1.5 (event-home handicap for Mexico = -1.5)
                    {"name": "Mexico", "point": -1.5, "price": 2.55}]}
            ]}
        ]
    }
    # build_constraints uses event home/away; event-home = South Africa
    # South Africa point 1.5 -> fixture-home (South Africa here) handicap = +1.5
    c = build_constraints(event)
    lines_before_swap = dict(c["spreads"])
    # The event-home (South Africa) has point +1.5 stored as home-covers line +1.5
    # This means "South Africa covers if SA_goals - MEX_goals > -1.5" (always true if draw or win)
    assert 1.5 in lines_before_swap

    # After swap flip (as done in predict.py when swapped=True):
    swapped_spreads = [(-line, p) for line, p in c["spreads"]]
    lines_after_swap = dict(swapped_spreads)
    # Now fixture-home = Mexico; Mexico's handicap was -1.5, so line = -1.5
    assert -1.5 in lines_after_swap
