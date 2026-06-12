import pytest

from model.predict import match_event, predict_match

CFG = {
    "pool": {"scoring": {"group": {"exact": 5, "gd": 3, "winner": 2},
                         "r32_r16": {"exact": 8, "gd": 5, "winner": 3},
                         "qf_plus": {"exact": 11, "gd": 7, "winner": 5}}},
    "elo": {"disagreement_threshold": 0.15},
    "ensemble": {"market": 1.0, "experts": 1.0},
}
FIXTURE = {"match_id": 1, "stage": "Group A", "home": "Mexico",
           "away": "South Africa", "venue": "Mexico City"}
ELO = {"mexico": 1800, "south africa": 1620}


def test_match_event_direct_and_swapped(sample_event):
    e, swapped = match_event(FIXTURE, [sample_event])
    assert e is sample_event and swapped is False
    flipped = dict(FIXTURE, home="South Africa", away="Mexico")
    e, swapped = match_event(flipped, [sample_event])
    assert e is sample_event and swapped is True
    assert match_event(dict(FIXTURE, home="Brazil"), [sample_event]) == (None, False)


def test_market_prediction(sample_event, sample_extras):
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG)
    assert p.source == "market"
    assert p.books_count == 2
    assert p.probs["home"] > p.probs["away"]
    assert p.pool1["score"] != p.pool2["score"]
    assert len(p.ep_table) == 5
    assert p.elo_disagrees is False  # Elo also makes Mexico favorite here


def test_swapped_orientation(sample_event):
    flipped = dict(FIXTURE, home="South Africa", away="Mexico")
    p = predict_match(flipped, sample_event, None, True, ELO, CFG)
    assert p.probs["away"] > p.probs["home"]  # Mexico is now the away side


def test_elo_fallback():
    p = predict_match(FIXTURE, None, None, False, ELO, CFG)
    assert p.source == "elo"
    assert "model-only" in p.note
    assert p.pool1 is not None


def test_no_data_at_all():
    p = predict_match(FIXTURE, None, None, False, {}, CFG)
    assert p.source == "none"
    assert p.pool1 is None


def test_stage_points():
    from model.predict import stage_points
    assert stage_points("Group A", CFG)["exact"] == 5
    assert stage_points("Knockout R4", CFG)["exact"] == 8
    assert stage_points("Knockout R5", CFG)["exact"] == 8
    assert stage_points("Knockout R6", CFG)["exact"] == 11
    assert stage_points("Knockout R8", CFG)["exact"] == 11
