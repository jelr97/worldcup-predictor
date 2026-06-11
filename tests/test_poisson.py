import numpy as np
import pytest

from model.poisson import outcome_probs, prob_btts, prob_over, score_matrix


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
