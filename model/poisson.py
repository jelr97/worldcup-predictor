"""Dixon-Coles-adjusted Poisson scoreline model."""
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
    g = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return float(m[g > line].sum())


def prob_btts(m):
    return float(m[1:, 1:].sum())
