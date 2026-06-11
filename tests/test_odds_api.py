import requests

from data.odds_api import OddsClient


def make_client(tmp_path):
    return OddsClient("k", "soccer_fifa_world_cup", cache_dir=tmp_path)


def test_cache_round_trip(tmp_path):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "e1"}])
    data, age = c.load_cache("main")
    assert data == [{"id": "e1"}]
    assert age < 0.01


def test_fresh_cache_skips_network(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "e1"}])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))
    data, age = c.get_main_odds(max_age_hours=12)
    assert data == [{"id": "e1"}]


def test_network_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "old"}])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    data, age = c.get_main_odds(max_age_hours=0)  # cache counts as stale
    assert data == [{"id": "old"}]


def test_no_cache_no_network_returns_none(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    data, age = c.get_main_odds()
    assert data is None and age is None


def test_corrupted_cache_is_ignored(tmp_path):
    c = make_client(tmp_path)
    (tmp_path / "main.json").write_text("{truncated", encoding="utf-8")
    assert c.load_cache("main") == (None, None)
