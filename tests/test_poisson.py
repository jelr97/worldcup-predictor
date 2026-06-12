import numpy as np
import pytest

from model.poisson import outcome_probs, prob_btts, prob_over, score_matrix
from model.poisson import prob_home_covers, solve_rates


def test_matrix_is_distribution():
    m = score_matrix(1.4, 1.1)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_dixon_coles_boosts_low_draws():
    dc = score_matrix(1.4, 1.1, rho=-0.10)
    plain = score_matrix(1.4, 1.1, rho=0.0)
    assert dc[0, 0] > plain[0, 0]
    assert dc[1, 1] > plain[1, 1]
    assert dc[1, 0] < plain[1, 0]


def test_outcome_probs():
    m = score_matrix(2.0, 0.8)
    oc = outcome_probs(m)
    assert sum(oc.values()) == pytest.approx(1.0)
    assert oc["home"] > oc["away"]


def test_over_and_btts():
    m = score_matrix(1.4, 1.1)
    g = np.add.outer(np.arange(11), np.arange(11))
    assert prob_over(m, 2.5) == pytest.approx(float(m[g >= 3].sum()))
    assert prob_btts(m) == pytest.approx(float(m[1:, 1:].sum()))
    assert 0 < prob_over(m, 2.5) < 1


def _constraints_from(lh, la):
    m = score_matrix(lh, la)
    return {
        "1x2": outcome_probs(m),
        "totals": [(2.5, prob_over(m, 2.5))],
        "btts": prob_btts(m),
    }


def test_round_trip_full_constraints():
    lh, la = solve_rates(_constraints_from(1.8, 0.9))
    assert lh == pytest.approx(1.8, abs=0.05)
    assert la == pytest.approx(0.9, abs=0.05)


def test_round_trip_1x2_only():
    m = score_matrix(1.5, 1.0)
    lh, la = solve_rates({"1x2": outcome_probs(m), "totals": [], "btts": None})
    assert lh == pytest.approx(1.5, abs=0.15)
    assert la == pytest.approx(1.0, abs=0.15)


def test_solver_floors_degenerate_rates():
    lh, la = solve_rates({"1x2": {"home": 0.97, "draw": 0.02, "away": 0.01},
                          "totals": [], "btts": None})
    assert la >= 0.01
    assert lh > 1.0


# ── prob_home_covers ──────────────────────────────────────────────────────────

def test_prob_home_covers_half_line():
    """home covers -1.5 iff i - j > 1.5, i.e., home wins by >= 2."""
    m = score_matrix(2.0, 0.8)
    p = prob_home_covers(m, -1.5)
    # Manually: sum cells where i - j >= 2
    g = np.subtract.outer(np.arange(11), np.arange(11))
    expected = float(m[g >= 2].sum())
    assert p == pytest.approx(expected)


def test_prob_home_covers_complement_half_line():
    """For a half-integer line, home_covers(L) + not_home_covers(L) == 1.0.

    On half-integer lines i-j is always an integer so P(i-j == -L) = 0;
    therefore P(i-j > -L) + P(i-j < -L) = 1 exactly.
    The 'away covers' probability = 1 - prob_home_covers(m, L).
    """
    m = score_matrix(1.5, 1.2)
    for line in (-1.5, -0.5, 0.5, 1.5):
        p_home = prob_home_covers(m, line)
        # Since -line is half-integer, i-j can never equal -line (integers),
        # so prob_home_covers exhausts the entire probability mass together
        # with its complement.
        g = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        p_away = float(m[g <= -line].sum())   # strictly < -line on half-int
        assert p_home + p_away == pytest.approx(1.0, abs=1e-9)


def test_prob_home_covers_strong_favorite():
    """Strong home side (lam_home >> lam_away) should cover -0.5 most of the time."""
    m = score_matrix(3.0, 0.5)
    p = prob_home_covers(m, -0.5)  # home wins by at least 1
    assert p > 0.85


def test_solve_rates_with_spreads_constraint():
    """Solver respects a spreads constraint in the loss function."""
    lh, la = 1.8, 0.7
    m0 = score_matrix(lh, la)
    p_covers = prob_home_covers(m0, -1.5)
    constraints = {
        "1x2": outcome_probs(m0),
        "totals": [(2.5, prob_over(m0, 2.5))],
        "btts": prob_btts(m0),
        "spreads": [(-1.5, p_covers)],
    }
    lh2, la2 = solve_rates(constraints)
    # Round-trip should converge close to originals
    assert lh2 == pytest.approx(lh, abs=0.1)
    assert la2 == pytest.approx(la, abs=0.1)
