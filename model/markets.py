"""Build solve_rates constraints from raw Odds-API event JSON."""
from data.team_names import same_team
from model import implied


def _h2h_odds(market, home, away):
    out = {}
    for o in market.get("outcomes", []):
        if o["name"].lower() == "draw":
            out["draw"] = o["price"]
        elif same_team(o["name"], home):
            out["home"] = o["price"]
        elif same_team(o["name"], away):
            out["away"] = o["price"]
    return out if len(out) == 3 else None


def _totals_odds(market):
    """-> {line: {'over': odds, 'under': odds}} for complete over/under pairs.

    Integer lines are skipped: they carry push semantics (stake refunded on
    exact total) that the Poisson over-probability does not model.
    """
    lines = {}
    for o in market.get("outcomes", []):
        line = o.get("point")
        side = o["name"].lower()
        if not isinstance(line, (int, float)) or line * 2 % 2 != 1:
            continue
        if side in ("over", "under"):
            lines.setdefault(line, {})[side] = o["price"]
    return {ln: v for ln, v in lines.items() if len(v) == 2}


def _spreads_odds(market, home, away):
    """-> {line: {'home': odds, 'away': odds}} for half-integer lines only.

    line is the fixture-home handicap (e.g. -1.5 means home -1.5).
    Outcomes carry the team name and a 'point' which is the handicap from
    that team's perspective (home outcome point = home_line; away outcome
    point = -home_line, so away_point = +1.5 when home_line = -1.5).
    We key the result on the fixture-home handicap.
    """
    lines = {}
    for o in market.get("outcomes", []):
        line = o.get("point")
        if not isinstance(line, (int, float)) or line * 2 % 2 != 1:
            continue   # skip integer lines (push semantics) and non-numeric
        name = o.get("name", "")
        if same_team(name, home):
            # The point on the home outcome IS the fixture-home handicap
            lines.setdefault(line, {})["home"] = o["price"]
        elif same_team(name, away):
            # The point on the away outcome is -home_line, so home_line = -line
            home_line = -line
            lines.setdefault(home_line, {})["away"] = o["price"]
    return {ln: v for ln, v in lines.items() if len(v) == 2}


def _btts_odds(market):
    out = {o["name"].lower(): o["price"] for o in market.get("outcomes", [])
           if o["name"].lower() in ("yes", "no")}
    return out if len(out) == 2 else None


def build_constraints(event, extras=None):
    """Returns a constraints dict for poisson.solve_rates, or None without 1X2.

    event: item from /sports/{sport}/odds (h2h, totals)
    extras: response from /sports/{sport}/events/{id}/odds (btts, alternate_totals)
    """
    home, away = event["home_team"], event["away_team"]
    h2h_books, totals_books, btts_books, spreads_books = {}, {}, {}, {}
    for src in [event] + ([extras] if extras else []):
        for book in src.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    d = implied.devig(_h2h_odds(mkt, home, away) or {})
                    if d:
                        h2h_books[book["key"]] = d
                elif mkt["key"] in ("totals", "alternate_totals"):
                    for line, ou in _totals_odds(mkt).items():
                        d = implied.devig(ou)
                        if d:
                            totals_books.setdefault(line, {})[book["key"]] = d
                elif mkt["key"] == "btts":
                    d = implied.devig(_btts_odds(mkt) or {})
                    if d:
                        btts_books[book["key"]] = d
                elif mkt["key"] == "spreads":
                    for line, ha in _spreads_odds(mkt, home, away).items():
                        d = implied.devig(ha)
                        if d:
                            spreads_books.setdefault(line, {})[book["key"]] = d
    one_x2 = implied.aggregate(h2h_books)
    if not one_x2:
        return None
    totals = []
    for line, books in sorted(totals_books.items()):
        agg = implied.aggregate(books)
        if agg:
            totals.append((line, agg["over"]))
    btts_agg = implied.aggregate(btts_books)
    spreads = []
    for line, books in sorted(spreads_books.items()):
        agg = implied.aggregate(books)
        if agg:
            spreads.append((line, agg["home"]))
    # When the odds event was swapped (home/away reversed vs fixture), the
    # caller flips 1x2 home<->away AND must flip spread lines AND complement
    # the cover probability: the line stored here is the event-home handicap;
    # when swapped, fixture-home = event-away, so fixture-home handicap =
    # -event_home_handicap, and P(fixture-home covers L) = 1 - P(event-home
    # covers -L). The caller does both flips at the same place it flips 1x2.
    return {"1x2": one_x2, "totals": totals,
            "btts": btts_agg["yes"] if btts_agg else None,
            "spreads": spreads,
            "books_count": len(h2h_books)}
