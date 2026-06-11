import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"
ET = ZoneInfo("America/New_York")


def load_fixtures(path=FIXTURES_PATH):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in raw:
        f = dict(f)
        f["kickoff_utc"] = datetime.fromisoformat(f["kickoff_utc"].replace("Z", "+00:00"))
        f["kickoff_et"] = f["kickoff_utc"].astimezone(ET)
        out.append(f)
    return sorted(out, key=lambda f: f["kickoff_utc"])


def upcoming(fixtures, now=None, window_hours=48):
    """Matches from 3h ago (today's in-play still visible) to now + window."""
    now = now or datetime.now(timezone.utc)
    end = now + timedelta(hours=window_hours)
    return [f for f in fixtures
            if now - timedelta(hours=3) <= f["kickoff_utc"] <= end]
