"""Canonical team-name matching across data sources (fixtures, odds API, Elo)."""
import unicodedata

# post-accent-strip, lowercase source name -> canonical name
ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "cote d'ivoire": "ivory coast",
    "turkiye": "turkey",
    "czechia": "czech republic",
    "bosnia and herzegovina": "bosnia",
    "bosnia-herzegovina": "bosnia",
    "cabo verde": "cape verde",
    "congo dr": "dr congo",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    return ALIASES.get(s, s)


def same_team(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)
