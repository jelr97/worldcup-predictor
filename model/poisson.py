"""Dixon-Coles-adjusted Poisson scoreline model."""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as _poisson

RHO = -0.10   # fixed low-score correlation; typical fitted range -0.05..-0.15
MAX_GOALS = 10


def score_matrix(lam_home, lam_away, rho=RHO):
    """P(home goals=i, away goals=j) for i,j in 0..MAX_GOALS. Rows = home."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(_poisson.pmf(g, lam_home), _poisson.pmf(g, lam_away))
    # Dixon-Coles tau adjustment on the four low-score cells
    m[0, 0] *= 1 - lam_home * lam_away * rho
    m[0, 1] *= 1 + lam_home * rho
    m[1, 0] *= 1 + lam_away * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(m):
    return {
        "home": float(np.tril(m, -1).sum()),   # i > j
        "draw": float(np.trace(m)),
        "away": float(np.triu(m, 1).sum()),    # j > i
    }


def prob_over(m, line):
    """P(total goals > line). Use half-integer lines (X.5) only: for integer
    lines the push mass (total == line) is excluded, not redistributed."""
    g = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return float(m[g > line].sum())


def prob_btts(m):
    return float(m[1:, 1:].sum())


def prob_home_covers(m, line):
    """P(home covers the handicap) = mass where (i - j) > -line.

    line is the fixture-home handicap (NEGATIVE means home gives goals: e.g.
    line=-1.5 means home must win by >1.5 to cover; line=+0.5 covers on a
    home win or draw).
    Home covers iff (i - j) + line > 0  ⟺  (i - j) > -line.
    Use half-integer lines only (push-free).
    """
    g = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return float(m[g > -line].sum())


def solve_rates(constraints, total_prior=2.5, rho=RHO):
    """Find (lam_home, lam_away) whose matrix best reproduces market probs.

    constraints: {'1x2': {'home','draw','away'} (required),
                  'totals': [(line, p_over), ...],
                  'btts': p_yes or None}
    Extra keys are ignored. When no totals are quoted, a soft prior keeps
    total goals near total_prior.
    """
    one_x2 = constraints["1x2"]
    totals = constraints.get("totals") or []
    btts = constraints.get("btts")
    spreads = constraints.get("spreads") or []

    def loss(x):
        lh, la = np.exp(x)
        m = score_matrix(lh, la, rho)
        oc = outcome_probs(m)
        err = sum((oc[k] - one_x2[k]) ** 2 for k in ("home", "draw", "away"))
        for line, p in totals:
            err += (prob_over(m, line) - p) ** 2
        if btts is not None:
            err += (prob_btts(m) - btts) ** 2
        for line, p in spreads:
            err += (prob_home_covers(m, line) - p) ** 2
        if not totals:
            err += 0.01 * ((lh + la) - total_prior) ** 2
        return err

    edge = one_x2["home"] - one_x2["away"]
    share = min(0.8, max(0.2, 0.5 + 0.4 * edge))
    x0 = np.log([total_prior * share, total_prior * (1 - share)])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
    if not res.success:
        warnings.warn(f"solve_rates did not converge: {res.message}")
    lh, la = np.exp(res.x)
    return max(float(lh), 0.01), max(float(la), 0.01)
