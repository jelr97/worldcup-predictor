"""The Odds API v4 client with on-disk JSON cache and quota tracking."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
DEFAULT_CACHE = Path(__file__).parent / "cache"


class OddsClient:
    def __init__(self, api_key, sport_key, regions=("uk", "eu"), cache_dir=DEFAULT_CACHE):
        self.api_key = api_key
        self.sport_key = sport_key
        self.regions = ",".join(regions)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_remaining = None

    def _get(self, path, **params):
        params = {"apiKey": self.api_key, "regions": self.regions,
                  "oddsFormat": "decimal", **params}
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        self.quota_remaining = r.headers.get("x-requests-remaining")
        return r.json()

    def save_cache(self, name, data):
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                   "quota_remaining": self.quota_remaining, "data": data}
        (self.cache_dir / f"{name}.json").write_text(json.dumps(payload),
                                                     encoding="utf-8")

    def load_cache(self, name):
        p = self.cache_dir / f"{name}.json"
        if not p.exists():
            return None, None
        payload = json.loads(p.read_text(encoding="utf-8"))
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(payload["fetched_at"])).total_seconds() / 3600
        self.quota_remaining = self.quota_remaining or payload.get("quota_remaining")
        return payload["data"], age_h

    def _cached_or_fetch(self, name, path, markets, max_age_hours, force):
        data, age = self.load_cache(name)
        if data is not None and not force and age <= max_age_hours:
            return data, age
        try:
            fresh = self._get(path, markets=markets)
            self.save_cache(name, fresh)
            return fresh, 0.0
        except requests.RequestException:
            return (data, age) if data is not None else (None, None)

    def get_main_odds(self, max_age_hours=12, force=False):
        """All upcoming WC events with h2h + totals. Returns (events, age_hours)."""
        return self._cached_or_fetch(
            "main", f"/sports/{self.sport_key}/odds", "h2h,totals",
            max_age_hours, force)

    def get_event_extras(self, event_id, max_age_hours=12, force=False):
        """BTTS + alternate totals for one event (extra API credits)."""
        return self._cached_or_fetch(
            f"event_{event_id}",
            f"/sports/{self.sport_key}/events/{event_id}/odds",
            "btts,alternate_totals", max_age_hours, force)
