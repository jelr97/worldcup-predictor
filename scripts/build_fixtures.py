"""One-time: build data/fixtures.json from fixturedownload.com."""
import json
import sys
from pathlib import Path

import requests

FEED = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures.json"


def main():
    r = requests.get(FEED, timeout=30)
    if r.status_code != 200:
        sys.exit(
            f"Feed returned {r.status_code}. Open https://fixturedownload.com, "
            "find the FIFA World Cup 2026 page, and update FEED with the exact "
            "json feed URL shown there."
        )
    fixtures = []
    for e in r.json():
        fixtures.append({
            "match_id": e["MatchNumber"],
            "stage": e.get("Group") or f"Knockout R{e['RoundNumber']}",
            "home": e["HomeTeam"],
            "away": e["AwayTeam"],
            "kickoff_utc": e["DateUtc"].replace(" ", "T").replace("Z", "") + "Z",
            "venue": e.get("Location", ""),
        })
    teams = {f["home"] for f in fixtures} | {f["away"] for f in fixtures}
    print(f"{len(fixtures)} matches, {len(teams)} team slots "
          "(knockouts may be placeholders like '1A')")
    OUT.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
