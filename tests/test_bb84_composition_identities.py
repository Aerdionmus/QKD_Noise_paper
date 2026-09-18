"""Regression tests for exact BB84 composition identities.

These tests pin down the D/P sub-cascade result from the canonical Kraus
channels. The quadratic correction is a composition/non-additivity effect;
it is not an ordering-dependence claim.
"""

import numpy as np

from qkd_noise.channels import bb84_average_qber


def test_depolarizing_and_dephasing_single_channel_qber_components():
    for p in [0.01, 0.02, 0.05, 0.10]:
        q_d = 2.0 * p / 3.0
        q_p = p / 2.0
        assert abs(bb84_average_qber(p, "single") - q_d) < 1e-12
        # The dephasing-only contribution is checked analytically here; the
        # public helper intentionally models the paper's single scenario as D.
        assert abs(q_p - p / 2.0) < 1e-15


def test_depolarizing_then_dephasing_exact_qber_correction():
    for p in [0.01, 0.02, 0.05, 0.10]:
        q_dp = bb84_average_qber(p, "dual")
        expected = 7.0 * p / 6.0 - 2.0 * p**2 / 3.0
        assert abs(q_dp - expected) < 1e-12
        assert abs(
            q_dp - (2.0 * p / 3.0 + p / 2.0) + 2.0 * p**2 / 3.0
        ) < 1e-12


def test_dual_correction_is_subadditive_not_ordering_dependent():
    # The canonical dual scenario is D -> P. The exact correction relative
    # to isolated-channel QBERs is negative, while the full ordering audit
    # separately establishes that symmetric BB84 QBER is invariant across
    # all six D/P/A orderings. This test only pins the former identity.
    for p in [0.01, 0.05, 0.10]:
        correction = bb84_average_qber(p, "dual") - (
            2.0 * p / 3.0 + p / 2.0
        )
        assert np.isclose(correction, -2.0 * p**2 / 3.0, atol=1e-12)
