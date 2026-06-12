"""Pure scoring function for a (pick, actual, stage) triple.

Three-tier polla scoring read from config:
  - exact score: full points
  - correct goal difference (or any draw when a draw was predicted): partial
  - correct winner only: fewest points

This module is importable from both model/ and scripts/ without side effects.
"""


def score_pick(pick: str, actual: str, pts: dict) -> int:
    """Score a pick string against an actual result string.

    Parameters
    ----------
    pick:   "H-A" format, e.g. "2-1"
    actual: "H-A" format, e.g. "2-1"
    pts:    {'exact': int, 'gd': int, 'winner': int} from config pool.scoring

    Returns
    -------
    Points earned: pts['exact'], pts['gd'], pts['winner'], or 0.
    """
    ph, pa = _parse(pick)
    ah, aa = _parse(actual)

    if ph == ah and pa == aa:
        return pts["exact"]

    pick_gd = ph - pa
    actual_gd = ah - aa

    if pick_gd == actual_gd:
        # Same goal difference: full gd credit.
        # Special case: draw prediction (gd=0) matches any draw — also covered
        # since actual_gd == 0 == pick_gd when both are draws.
        return pts["gd"]

    # Both predicted a draw and the result was a draw with a different GD?
    # Not possible — draws all have GD=0. So just check correct winner.
    pick_winner = _winner(ph, pa)
    actual_winner = _winner(ah, aa)
    if pick_winner == actual_winner:
        return pts["winner"]

    return 0


def _parse(score: str) -> tuple[int, int]:
    h, a = score.split("-")
    return int(h), int(a)


def _winner(h: int, a: int) -> str:
    if h > a:
        return "home"
    if a > h:
        return "away"
    return "draw"
