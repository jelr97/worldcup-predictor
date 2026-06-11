import pytest

from model.predict import match_event, predict_match

CFG = {
    "pool": {"pts_exact": 3, "pts_outcome": 1},
    "elo": {"disagreement_threshold": 0.15},
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
