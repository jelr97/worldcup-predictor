"""Tests for model/experts.py: pick_to_rates and experts_matrix."""
import numpy as np
import pytest

from model.experts import experts_matrix, pick_to_rates
from model.poisson import score_matrix

_RATE_CAP = 6.0
_RATE_BUMP = 0.4


# ── pick_to_rates ─────────────────────────────────────────────────────────────

def test_pick_to_rates_basic():
    lh, la = pick_to_rates((2, 1))
    assert lh == pytest.approx(2.4)
    assert la == pytest.approx(1.4)


def test_pick_to_rates_zero():
    lh, la = pick_to_rates((0, 0))
    assert lh == pytest.approx(0.4)
    assert la == pytest.approx(0.4)


def test_pick_to_rates_cap():
    """Picks >= 5.6 should be capped at 6.0."""
    lh, la = pick_to_rates((7, 6))
    assert lh == _RATE_CAP
    assert la == _RATE_CAP


def test_pick_to_rates_cap_exact():
    """Pick of 6 -> 6.0 (6 + 0.4 = 6.4 > 6.0, so capped)."""
    lh, _ = pick_to_rates((6, 0))
    assert lh == _RATE_CAP


def test_pick_to_rates_modal_cell():
    """The predicted score should be the modal cell of the resulting Poisson matrix.

    For Poisson(lambda), the mode is floor(lambda) for lambda >= 1.
    With bump=0.4, pick (2,1) -> rates (2.4, 1.4) -> modes floor(2.4)=2, floor(1.4)=1.
    So cell (2,1) should be the maximum.
    """
    pick = (2, 1)
    lh, la = pick_to_rates(pick)
    m = score_matrix(lh, la)
    # Find modal cell
    modal = np.unravel_index(np.argmax(m), m.shape)
    assert modal == pick


def test_pick_to_rates_modal_cell_zero():
    """Pick (0,0) -> rates (0.4, 0.4) -> mode should be (0,0)."""
    pick = (0, 0)
    lh, la = pick_to_rates(pick)
    m = score_matrix(lh, la)
    modal = np.unravel_index(np.argmax(m), m.shape)
    assert modal == pick


def test_pick_to_rates_modal_cell_asymmetric():
    """Pick (3,0) -> rates (3.4, 0.4) -> mode should be (3,0)."""
    pick = (3, 0)
    lh, la = pick_to_rates(pick)
    m = score_matrix(lh, la)
    modal = np.unravel_index(np.argmax(m), m.shape)
    assert modal == pick


# ── experts_matrix ────────────────────────────────────────────────────────────

def test_experts_matrix_sums_to_one():
    picks = {"davo": (2, 0), "maldini": (0, 2)}
    m = experts_matrix(picks)
    assert m.sum() == pytest.approx(1.0, abs=1e-8)


def test_experts_matrix_disagreement():
    """Davo 2-0 + Maldini 0-2 -> P(2-0) and P(0-2) each > P(1-1)."""
    picks = {"davo": (2, 0), "maldini": (0, 2)}
    m = experts_matrix(picks)
    # P(2-0) should be > P(1-1)
    assert m[2, 0] > m[1, 1], f"P(2-0)={m[2,0]:.4f} should > P(1-1)={m[1,1]:.4f}"
    # P(0-2) should be > P(1-1)
    assert m[0, 2] > m[1, 1], f"P(0-2)={m[0,2]:.4f} should > P(1-1)={m[1,1]:.4f}"


def test_experts_matrix_agreement():
    """When both experts agree, the result should clearly favor that score."""
    picks = {"davo": (2, 1), "maldini": (2, 1)}
    m = experts_matrix(picks)
    modal = np.unravel_index(np.argmax(m), m.shape)
    assert modal == (2, 1)


def test_experts_matrix_shape():
    picks = {"davo": (1, 0), "maldini": (2, 1)}
    m = experts_matrix(picks)
    assert m.shape == (11, 11)  # MAX_GOALS+1 = 11


def test_experts_matrix_nonnegative():
    picks = {"davo": (1, 2), "maldini": (0, 1)}
    m = experts_matrix(picks)
    assert (m >= 0).all()
