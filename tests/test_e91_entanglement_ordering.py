"""
Tests for the E91 entanglement (concurrence) ordering study
(scripts/run_e91_entanglement_ordering_study.py).

Verifies the concurrence closed form derived for the two-sided |Phi+>
output -- C(rho) = max(0, M_x^2 - Q) -- against the full numerical
Wootters concurrence, the [0,1] range, the ideal/zero-noise limits, and
the ordering-class comparison, without assuming in advance whether
concurrence is or is not ordering-sensitive.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOL = 1e-12


def _load_module(name, relpath):
    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module("run_e91_entanglement_ordering_study", "scripts/run_e91_entanglement_ordering_study.py")


@pytest.fixture(scope="module")
def audit(mod):
    return mod.AUDIT


# ------------------------------------------------------------------
# 1-3. Concurrence range, ideal state, zero-noise output
# ------------------------------------------------------------------

def test_concurrence_range_bounded(mod, audit):
    for d, q, gamma in [(0.01, 0.0, 0.02), (0.1, 0.1, 0.1), (0.5, 0.0, 0.5), (0.9, 0.05, 0.9)]:
        for cls in (1, 2):
            point = mod.evaluate_and_verify(d, q, gamma, cls)
            assert -1e-9 <= point["concurrence"] <= 1.0 + 1e-9


def test_concurrence_of_ideal_phi_plus_is_one(mod, audit):
    rho = audit.phi_plus()
    C = mod.concurrence_wootters(rho)
    assert abs(C - 1.0) < 1e-9


def test_zero_noise_output_has_concurrence_one(mod, audit):
    rho0 = audit.phi_plus()
    for name, seq in audit.ORDERINGS.items():
        rho = audit.apply_ordering(rho0, seq, 0.0, 0.0, 0.0, "two")
        C = mod.concurrence_wootters(rho)
        assert abs(C - 1.0) < 1e-9


# ------------------------------------------------------------------
# 4-6. Hermiticity, trace, positivity (of the underlying output states)
# ------------------------------------------------------------------

def test_output_states_physical(audit):
    rho0 = audit.phi_plus()
    for d, q, gamma in [(0.02, 0.01, 0.03), (0.2, 0.1, 0.3)]:
        for cls in (1, 2):
            name = "A o P o D" if cls == 1 else "P o D o A"
            rho = audit.apply_ordering(rho0, audit.ORDERINGS[name], d, q, gamma, "two")
            assert abs(np.trace(rho) - 1.0) < 1e-9
            assert np.max(np.abs(rho - rho.conj().T)) < 1e-9
            eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
            assert np.min(eigvals) > -1e-9


# ------------------------------------------------------------------
# 7. Analytic-vs-direct (closed-form vs Wootters) concurrence agreement
# ------------------------------------------------------------------

def test_concurrence_closed_form_matches_wootters(mod, audit):
    """The derived X-state closed form C=max(0, M_x^2 - Q) must match
    the full, general Wootters concurrence formula, across a range of
    parameters spanning both the entangled and (if reached) separable
    regimes."""
    rho0 = audit.phi_plus()
    max_diff = 0.0
    for d in (0.01, 0.05, 0.1, 0.3, 0.6):
        for q in (0.0, 0.05, 0.2):
            for gamma in (0.01, 0.1, 0.4, 0.8):
                for cls, name in ((1, "A o P o D"), (2, "P o D o A")):
                    rho = audit.apply_ordering(rho0, audit.ORDERINGS[name], d, q, gamma, "two")
                    T = audit.correlation_tensor(rho)
                    C_closed = mod.concurrence_closed_form(T[0, 0], T[2, 2])
                    C_numeric = mod.concurrence_wootters(rho)
                    max_diff = max(max_diff, abs(C_closed - C_numeric))
    assert max_diff < 1e-9


def test_x_state_has_zero_01_10_coherence(audit):
    """Confirms the derived structural claim (not assumed): the
    two-sided output has exactly zero (|01>,|10>) coherence, which is
    what permits the simplified closed-form concurrence."""
    rho0 = audit.phi_plus()
    rho = audit.apply_ordering(rho0, audit.ORDERINGS["A o P o D"], 0.05, 0.03, 0.07, "two")
    assert abs(rho[1, 2]) < TOL
    assert abs(rho[2, 1]) < TOL


# ------------------------------------------------------------------
# 8. Ordering-class comparison
# ------------------------------------------------------------------

def test_concurrence_difference_matches_half_delta_Tzz_when_unclipped(mod, audit):
    """In the low-noise (unclipped) regime, Delta C should equal
    exactly Delta T_zz / 2 -- derived from C=max(0,Mx^2-Q) when neither
    class is clipped to zero."""
    for d, gamma in [(0.01, 0.01), (0.05, 0.05), (0.02, 0.08)]:
        q = d  # shared-style point, arbitrary but small
        point1 = mod.evaluate_and_verify(d, q, gamma, 1)
        point2 = mod.evaluate_and_verify(d, q, gamma, 2)
        if point1["concurrence"] > 0 and point2["concurrence"] > 0:
            delta_c = point1["concurrence"] - point2["concurrence"]
            delta_Tzz = audit.predicted_delta_Tzz(d, gamma)
            assert abs(delta_c - delta_Tzz / 2) < TOL


def test_concurrence_ordering_sensitivity_in_shared_p_low_noise(mod):
    """Direct empirical test (not assumed in either direction): at
    small shared p, do the two ordering classes give different
    concurrence?"""
    p = 0.02
    point1 = mod.evaluate_and_verify(p, p, p, 1)
    point2 = mod.evaluate_and_verify(p, p, p, 2)
    # This assertion records the actual observed behavior; if it ever
    # fails, that is new information about the model, not a bug to
    # silently paper over.
    assert abs(point1["concurrence"] - point2["concurrence"]) > 1e-9


# ------------------------------------------------------------------
# 9. Shared-p regression points
# ------------------------------------------------------------------

def test_shared_p_regression_points(mod):
    expected = {
        0.001: None,  # not hard-coded to a specific value; just checked for consistency below
    }
    for p in (0.001, 0.01, 0.02, 0.05):
        point1 = mod.evaluate_and_verify(p, p, p, 1)
        point2 = mod.evaluate_and_verify(p, p, p, 2)
        assert 0.0 <= point1["concurrence"] <= 1.0
        assert 0.0 <= point2["concurrence"] <= 1.0
        # Class 1 (t_z=gamma, the larger translation) should have
        # T_zz1 >= T_zz2, hence concurrence1 >= concurrence2 in the
        # unclipped regime -- checked directly, not assumed.
        assert point1["concurrence"] >= point2["concurrence"] - 1e-12


def test_esd_thresholds_are_calculated_and_class_dependent(mod):
    esd1 = mod.esd_threshold(1)
    esd2 = mod.esd_threshold(2)
    assert esd1 is not None and esd2 is not None
    assert 0.0 < esd1 < 1.0
    assert 0.0 < esd2 < 1.0
    # The two classes need not (and, per the derivation, generally do
    # not) share an ESD threshold -- checked, not assumed.
    assert abs(esd1 - esd2) > 1e-6


# ------------------------------------------------------------------
# 10. No regression of existing functionality (imports cleanly, reuses
#     canonical channels without duplication)
# ------------------------------------------------------------------

def test_does_not_redefine_canonical_channels(mod, audit):
    from qkd_noise.channels import (
        amplitude_damping_kraus,
        apply_channel,
        dephasing_kraus,
        depolarizing_kraus,
    )

    assert audit.depolarizing_kraus is depolarizing_kraus
    assert audit.dephasing_kraus is dephasing_kraus
    assert audit.amplitude_damping_kraus is amplitude_damping_kraus
    assert audit.apply_channel is apply_channel


def test_concurrence_never_negative_before_clipping_interpretation(mod):
    """The raw (unclipped) argument M_x^2 - Q can be negative (that IS
    the separable regime); concurrence itself must never report a
    negative value regardless."""
    for p in (0.3, 0.5, 0.8, 0.95):
        for cls in (1, 2):
            point = mod.evaluate_and_verify(p, p, p, cls)
            assert point["concurrence"] >= 0.0
