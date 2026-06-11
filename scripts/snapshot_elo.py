"""Refresh data/elo_snapshot.csv from eloratings.net. Re-run every few days
during the tournament (ratings move after each match day).

The site serves two TSV files:
  - en.teams.tsv : 2-letter code -> canonical country name (tab-separated, first name used)
  - World.tsv    : ranked rows with fields [rank, prev_rank, code, rating, ...]

We join them to produce team,rating rows.
"""
import csv
import sys
from pathlib import Path

import requests

TEAMS_URL = "https://www.eloratings.net/en.teams.tsv"
RATINGS_URL = "https://www.eloratings.net/World.tsv"
OUT = Path(__file__).resolve().parents[1] / "data" / "elo_snapshot.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url):
    r = requests.get(url, timeout=30, headers=HEADERS)
    if r.status_code != 200:
        sys.exit(
            f"{url} returned {r.status_code}.\n"
            "Manual fallback: open https://www.eloratings.net, copy the ranking "
            "table (at least the 48 WC teams) into data/elo_snapshot.csv with "
            "header 'team,rating'."
        )
    return r.text


def parse_teams(text):
    """Return dict: 2-letter code -> canonical English name."""
    code_to_name = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            code = parts[0].strip()
            name = parts[1].strip()
            # Skip _loc suffix entries (localisation variants)
            if not code.endswith("_loc") and name:
                code_to_name[code] = name
    return code_to_name


def parse_ratings(text, code_to_name):
    """Return list of {team, rating} dicts, ordered as they appear (descending rating)."""
    rows = []
    for line in text.splitlines():
        fields = line.split("\t")
        # Expected layout: rank, prev_rank, code, rating, ...
        if len(fields) < 4:
            continue
        code = fields[2].strip()
        try:
            rating = int(fields[3])
        except ValueError:
            continue
        if 1000 <= rating <= 2400 and code in code_to_name:
            rows.append({"team": code_to_name[code], "rating": rating})
    return rows


def main():
    code_to_name = parse_teams(fetch(TEAMS_URL))
    rows = parse_ratings(fetch(RATINGS_URL), code_to_name)

    if len(rows) < 50:
        sys.exit(
            f"Only parsed {len(rows)} rows — source format may have changed. "
            "Inspect the response, adjust parsing, or use the manual fallback "
            "described above."
        )

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "rating"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} ratings; top 5: {[r['team'] for r in rows[:5]]}")


if __name__ == "__main__":
    main()
