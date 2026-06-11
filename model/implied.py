"""Bookmaker odds -> de-vigged probabilities, aggregated across books."""
import statistics


def devig(odds):
    """{'home': 2.1, 'draw': 3.3, 'away': 3.6} -> probs summing to 1, or None."""
    if not odds or any(o is None or o <= 1.0 for o in odds.values()):
        return None
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def aggregate(per_book):
    """per_book: {bookmaker_key: {outcome: prob}} -> consensus probs.

    Median across books; when Pinnacle quotes the match:
    0.5 * pinnacle + 0.5 * median(other books). Renormalized.
    Books are aggregated over their shared outcome keys.
    """
    per_book = {b: p for b, p in per_book.items() if p}
    if not per_book:
        return None
    keys = set.intersection(*[set(p) for p in per_book.values()])
    if not keys:
        return None
    pinn = per_book.get("pinnacle")
    others = {b: p for b, p in per_book.items() if b != "pinnacle"}
    med = ({k: statistics.median(p[k] for p in others.values()) for k in keys}
           if others else None)
    if pinn and med:
        probs = {k: 0.5 * pinn[k] + 0.5 * med[k] for k in keys}
    else:
        probs = {k: (pinn or med)[k] for k in keys}
    total = sum(probs.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in probs.items()}
