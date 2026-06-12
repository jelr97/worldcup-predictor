import json
import math
from datetime import datetime, timedelta, timezone

import requests

from data.odds_api import OddsClient


def make_client(tmp_path):
    return OddsClient("k", "soccer_fifa_world_cup", cache_dir=tmp_path)


def write_aged_cache(tmp_path, name, data, hours_old):
    payload = {"fetched_at": (datetime.now(timezone.utc)
                              - timedelta(hours=hours_old)).isoformat(),
               "quota_remaining": "400", "data": data}
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


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


def test_infinite_max_age_serves_ancient_cache_without_network(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    write_aged_cache(tmp_path, "main", [{"id": "old"}], hours_old=72)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))
    data, age = c.get_main_odds(max_age_hours=math.inf)
    assert data == [{"id": "old"}]
    assert age > 71


def test_force_skips_network_when_cache_is_minutes_old(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "fresh"}])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))
    data, age = c.get_main_odds(max_age_hours=math.inf, force=True)
    assert data == [{"id": "fresh"}]


class _FakeResp:
    headers = {"x-requests-remaining": "399"}

    @staticmethod
    def raise_for_status():
        pass

    @staticmethod
    def json():
        return [{"id": "new"}]


def test_force_refetches_old_cache(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    write_aged_cache(tmp_path, "main", [{"id": "old"}], hours_old=3)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp())
    data, age = c.get_main_odds(max_age_hours=math.inf, force=True)
    assert data == [{"id": "new"}]
    assert age == 0.0
    assert c.quota_remaining == "399"


def test_extras_force_floor_is_one_hour(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    write_aged_cache(tmp_path, "event_x", [{"id": "30min"}], hours_old=0.5)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))
    data, age = c.get_event_extras("x", max_age_hours=math.inf, force=True)
    assert data == [{"id": "30min"}]  # 30 min old: under the 1h floor, no refetch
