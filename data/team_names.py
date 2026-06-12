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
    "bosnia & herzegovina": "bosnia",
    "cabo verde": "cape verde",
    "congo dr": "dr congo",
    # ── Spanish-name aliases (from Poia.xlsx experts sheet) ───────────────────
    "alemania": "germany",
    "arabia saudita": "saudi arabia",
    "argelia": "algeria",
    "belgica": "belgium",
    "bosnia y herzegovina": "bosnia",
    "brasil": "brazil",
    "chequia": "czech republic",
    "corea del sur": "south korea",
    "costa de marfil": "ivory coast",
    "croacia": "croatia",
    "curazao": "curacao",
    "egipto": "egypt",
    "escocia": "scotland",
    "espana": "spain",
    "estados unidos": "united states",
    "francia": "france",
    "inglaterra": "england",
    "irak": "iraq",
    "japon": "japan",
    "jordania": "jordan",
    "marruecos": "morocco",
    "noruega": "norway",
    "nueva zelanda": "new zealand",
    "paises bajos": "netherlands",
    "rd congo": "dr congo",
    "sudafrica": "south africa",
    "suecia": "sweden",
    "suiza": "switzerland",
    "tunez": "tunisia",
    "turquia": "turkey",
    "uzbekistan": "uzbekistan",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    return ALIASES.get(s, s)


def same_team(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)
