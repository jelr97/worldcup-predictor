"""Flag emoji for each World Cup 2026 group-stage team.

Keys are normalized team names (via data.team_names.normalize).
"""
from data.team_names import normalize as _normalize

# All 48 group-stage teams, keyed by their normalized name.
FLAGS: dict[str, str] = {
    "algeria":       "🇩🇿",
    "argentina":     "🇦🇷",
    "australia":     "🇦🇺",
    "austria":       "🇦🇹",
    "belgium":       "🇧🇪",
    "bosnia":        "🇧🇦",
    "brazil":        "🇧🇷",
    "cape verde":    "🇨🇻",
    "canada":        "🇨🇦",
    "colombia":      "🇨🇴",
    "dr congo":      "🇨🇩",
    "croatia":       "🇭🇷",
    "curacao":       "🇨🇼",
    "czech republic":"🇨🇿",
    "ivory coast":   "🇨🇮",
    "ecuador":       "🇪🇨",
    "egypt":         "🇪🇬",
    "england":       "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "france":        "🇫🇷",
    "germany":       "🇩🇪",
    "ghana":         "🇬🇭",
    "haiti":         "🇭🇹",
    "iran":          "🇮🇷",
    "iraq":          "🇮🇶",
    "japan":         "🇯🇵",
    "jordan":        "🇯🇴",
    "south korea":   "🇰🇷",
    "mexico":        "🇲🇽",
    "morocco":       "🇲🇦",
    "netherlands":   "🇳🇱",
    "new zealand":   "🇳🇿",
    "norway":        "🇳🇴",
    "panama":        "🇵🇦",
    "paraguay":      "🇵🇾",
    "portugal":      "🇵🇹",
    "qatar":         "🇶🇦",
    "saudi arabia":  "🇸🇦",
    "scotland":      "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "senegal":       "🇸🇳",
    "south africa":  "🇿🇦",
    "spain":         "🇪🇸",
    "sweden":        "🇸🇪",
    "switzerland":   "🇨🇭",
    "tunisia":       "🇹🇳",
    "turkey":        "🇹🇷",
    "united states": "🇺🇸",
    "uruguay":       "🇺🇾",
    "uzbekistan":    "🇺🇿",
}


def flag(team: str) -> str:
    """Return the flag emoji for a team name, or '' if unknown."""
    return FLAGS.get(_normalize(team), "")
