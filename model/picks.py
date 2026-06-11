"""Rank scorelines by expected pool points; pick #1 for Pool 1, #2 for Pool 2."""
import numpy as np

from model.poisson import outcome_probs


def outcome_of(i, j):
    return "home" if i > j else "away" if j > i else "draw"


def expected_points(matrix, pts):
    """pts: {'exact': A, 'gd': B, 'winner': C} — three-tier polla scoring.

    Tiers per prediction (i, j): exact score; right goal difference (any draw
    counts for draw predictions); right winner only (empty tier for draws).
    """
    oc = outcome_probs(matrix)
    n = matrix.shape[0]
    g = np.subtract.outer(np.arange(n), np.arange(n))
    gd_prob = {d: float(matrix[g == d].sum()) for d in range(-(n - 1), n)}
    rows = []
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            p_gd = gd_prob[i - j]
            ep = p * pts["exact"] + (p_gd - p) * pts["gd"]
            outcome = outcome_of(i, j)
            if outcome != "draw":
                ep += (oc[outcome] - p_gd) * pts["winner"]
            rows.append({"score": f"{i}-{j}", "home_goals": i, "away_goals": j,
                         "p_exact": p, "ep": ep})
    # ties: higher exact probability, then fewer total goals (deterministic)
    rows.sort(key=lambda r: (-round(r["ep"], 9), -round(r["p_exact"], 9),
                             r["home_goals"] + r["away_goals"]))
    return rows


def top_picks(matrix, pts, n=5):
    rows = expected_points(matrix, pts)
    return {"pool1": rows[0], "pool2": rows[1], "table": rows[:n]}
