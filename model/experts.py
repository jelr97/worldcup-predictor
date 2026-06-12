"""Expert-prediction model layer.

pick_to_rates((a, b)):
    Converts an expert's exact-score pick to Poisson rate parameters.
    The +0.4 shift makes the predicted score the modal scoreline of the
    resulting Poisson distribution while keeping realistic probability mass
    on neighbouring scores.  The 6.0 cap tames extreme picks like 7-0.

experts_matrix(picks):
    Builds a scoreline probability matrix as the 0.5/0.5 average of the
    Dixon-Coles matrices implied by Davo's and Maldini's picks.
    Averaging distributions (not scores) preserves disagreement mass:
    Davo 2-0 + Maldini 0-2 keeps both scorelines more likely than 1-1.
"""
import numpy as np

from model.poisson import score_matrix

_RATE_BUMP = 0.4
_RATE_CAP = 6.0


def pick_to_rates(pick: tuple[int, int]) -> tuple[float, float]:
    """Convert an expert score pick to (lam_home, lam_away) Poisson rates.

    Parameters
    ----------
    pick: (home_goals, away_goals)

    Returns
    -------
    (lam_home, lam_away) each in (0, _RATE_CAP]
    """
    a, b = pick
    lam_home = min(a + _RATE_BUMP, _RATE_CAP)
    lam_away = min(b + _RATE_BUMP, _RATE_CAP)
    return lam_home, lam_away


def experts_matrix(picks: dict) -> np.ndarray:
    """Return the 0.5·DC-Davo + 0.5·DC-Maldini blended probability matrix.

    Parameters
    ----------
    picks: dict as returned by data.experts.picks_for — keys 'davo', 'maldini',
           each a (home_goals, away_goals) tuple.

    Returns
    -------
    numpy array of shape (MAX_GOALS+1, MAX_GOALS+1), normalised to sum 1.
    """
    davo_lh, davo_la = pick_to_rates(picks["davo"])
    maldini_lh, maldini_la = pick_to_rates(picks["maldini"])

    m_davo = score_matrix(davo_lh, davo_la)
    m_maldini = score_matrix(maldini_lh, maldini_la)

    blended = 0.5 * m_davo + 0.5 * m_maldini
    return blended / blended.sum()
