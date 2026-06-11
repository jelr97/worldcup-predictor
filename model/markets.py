"""Build solve_rates constraints from raw Odds-API event JSON."""
from data.team_names import same_team
from model import implied


def _h2h_odds(market, home, away):
    out = {}
    for o in market.get("outcomes", []):
        if o["name"] == "Draw":
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
        if line is None or line * 2 % 2 != 1:
            continue
        if side in ("over", "under"):
            lines.setdefault(line, {})[side] = o["price"]
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
    h2h_books, totals_books, btts_books = {}, {}, {}
    for src in [event] + ([extras] if extras else []):
        for book in src.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    odds = _h2h_odds(mkt, home, away)
                    if odds:
                        h2h_books[book["key"]] = implied.devig(odds)
                elif mkt["key"] in ("totals", "alternate_totals"):
                    for line, ou in _totals_odds(mkt).items():
                        d = implied.devig(ou)
                        if d:
                            totals_books.setdefault(line, {})[book["key"]] = d
                elif mkt["key"] == "btts":
                    d = implied.devig(_btts_odds(mkt) or {})
                    if d:
                        btts_books[book["key"]] = d
    one_x2 = implied.aggregate(h2h_books)
    if not one_x2:
        return None
    totals = []
    for line, books in sorted(totals_books.items()):
        agg = implied.aggregate(books)
        if agg:
            totals.append((line, agg["over"]))
    btts_agg = implied.aggregate(btts_books)
    return {"1x2": one_x2, "totals": totals,
            "btts": btts_agg["yes"] if btts_agg else None,
            "books_count": len(h2h_books)}
