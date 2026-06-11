"""Rank scorelines by expected pool points; pick #1 for Pool 1, #2 for Pool 2."""
from model.poisson import outcome_probs


def outcome_of(i, j):
    return "home" if i > j else "away" if j > i else "draw"


def expected_points(matrix, pts_exact, pts_outcome):
    oc = outcome_probs(matrix)
    rows = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            ep = p * pts_exact + (oc[outcome_of(i, j)] - p) * pts_outcome
            rows.append({"score": f"{i}-{j}", "home_goals": i, "away_goals": j,
                         "p_exact": p, "ep": ep})
    # ties: higher exact probability, then fewer total goals (deterministic)
    rows.sort(key=lambda r: (-round(r["ep"], 9), -round(r["p_exact"], 9),
                             r["home_goals"] + r["away_goals"]))
    return rows


def top_picks(matrix, pts_exact, pts_outcome, n=5):
    rows = expected_points(matrix, pts_exact, pts_outcome)
    return {"pool1": rows[0], "pool2": rows[1], "table": rows[:n]}
