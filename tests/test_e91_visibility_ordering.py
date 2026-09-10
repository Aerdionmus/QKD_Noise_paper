"""
Tests for the E91 Werner-state visibility robustness study
(scripts/run_e91_visibility_ordering_study.py).

Verifies the Werner-state construction, the derived v-dependent
correlation-tensor formulas, the v-independence of Delta T_zz, the
concurrence closed form, the fixed-CHSH identity, correct Q, and that
DIQKD strictly uses S_fixed (never S_max) -- without assuming any of
these survive at reduced visibility until checked.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOL = 1e-9


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_e91_visibility_ordering_study.py"
    spec = importlib.util.spec_from_file_location("run_e91_visibility_ordering_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def audit(mod):
    return mod.AUDIT


# ------------------------------------------------------------------
# Werner-state construction: normalization, positivity, limits
# ------------------------------------------------------------------

def test_werner_state_normalized_hermitian_positive(mod):
    for v in (0.0, 0.1, 0.5, 0.9, 1.0):
        rho = mod.werner_state(v)
        assert abs(np.trace(rho) - 1.0) < TOL
        assert np.max(np.abs(rho - rho.conj().T)) < TOL
        eigvals = np.linalg.eigvalsh(rho)
        assert np.min(eigvals) > -1e-10


def test_werner_state_v1_equals_phi_plus(mod, audit):
    rho = mod.werner_state(1.0)
    np.testing.assert_allclose(rho, audit.phi_plus(), atol=TOL)


def test_werner_state_v0_equals_maximally_mixed(mod):
    rho = mod.werner_state(0.0)
    np.testing.assert_allclose(rho, np.eye(4, dtype=complex) / 4, atol=TOL)


# ------------------------------------------------------------------
# Analytic T_xx, T_yy, T_zz formulas
# ------------------------------------------------------------------

def test_correlation_tensor_matches_v_dependent_formula(mod, audit):
    for d, q, gamma, v in [(0.02, 0.0, 0.03, 0.4), (0.06, 0.05, 0.08, 0.7), (0.1, 0.1, 0.1, 1.0)]:
        for cls, name in ((1, "A o P o D"), (2, "P o D o A")):
            rho_w = mod.werner_state(v)
            rho = audit.apply_ordering(rho_w, audit.ORDERINGS[name], d, q, gamma, "two")
            T = audit.correlation_tensor(rho)
            Mx = audit.predicted_Mx(d, q, gamma)
            Mz = audit.predicted_Mz(d, q, gamma)
            tz = audit.predicted_tz(d, q, gamma, cls)
            expected = np.diag([v * Mx**2, -v * Mx**2, v * Mz**2 + tz**2])
            np.testing.assert_allclose(T, expected, atol=TOL)


def test_marginals_unaffected_by_visibility(mod, audit):
    d, q, gamma = 0.05, 0.03, 0.07
    tz = audit.predicted_tz(d, q, gamma, 1)
    for v in (0.2, 0.6, 1.0):
        rho_w = mod.werner_state(v)
        rho = audit.apply_ordering(rho_w, audit.ORDERINGS["A o P o D"], d, q, gamma, "two")
        rA = audit.bloch_vector(audit.partial_trace_b(rho))
        rB = audit.bloch_vector(audit.partial_trace_a(rho))
        np.testing.assert_allclose(rA, [0, 0, tz], atol=TOL)
        np.testing.assert_allclose(rB, [0, 0, tz], atol=TOL)


# ------------------------------------------------------------------
# Delta T_zz is exactly v-independent
# ------------------------------------------------------------------

def test_delta_Tzz_independent_of_visibility(mod, audit):
    d, gamma = 0.04, 0.06
    predicted_delta = audit.predicted_delta_Tzz(d, gamma)
    values = set()
    for v in (0.1, 0.4, 0.7, 1.0):
        p1 = mod.evaluate_point(d, d, gamma, v, 1)
        p2 = mod.evaluate_point(d, d, gamma, v, 2)
        delta_observed = p1["T_zz"] - p2["T_zz"]
        assert abs(delta_observed - predicted_delta) < TOL
        values.add(round(delta_observed, 12))
    assert len(values) == 1


# ------------------------------------------------------------------
# Concurrence: Wootters vs closed form, for the Werner-state output
# ------------------------------------------------------------------

def test_concurrence_closed_form_matches_wootters_for_werner_output(mod, audit):
    max_diff = 0.0
    for d in (0.01, 0.05, 0.2):
        for gamma in (0.01, 0.1, 0.5):
            for v in (0.0, 0.3, 0.7, 1.0):
                for cls, name in ((1, "A o P o D"), (2, "P o D o A")):
                    rho_w = mod.werner_state(v)
                    rho = audit.apply_ordering(rho_w, audit.ORDERINGS[name], d, 0.02, gamma, "two")
                    T = audit.correlation_tensor(rho)
                    C_closed = mod.concurrence_closed_form(T[0, 0], T[2, 2])
                    C_numeric = mod.concurrence_wootters(rho)
                    max_diff = max(max_diff, abs(C_closed - C_numeric))
    assert max_diff < 1e-9


def test_werner_output_retains_zero_01_10_coherence(mod, audit):
    rho_w = mod.werner_state(0.5)
    rho = audit.apply_ordering(rho_w, audit.ORDERINGS["A o P o D"], 0.05, 0.03, 0.07, "two")
    assert abs(rho[1, 2]) < TOL
    assert abs(rho[2, 1]) < TOL


# ------------------------------------------------------------------
# Fixed-CHSH identity
# ------------------------------------------------------------------

def test_fixed_chsh_identity_holds_for_werner_output(mod, audit):
    for v in (0.3, 0.6, 1.0):
        point = mod.evaluate_point(0.03, 0.02, 0.04, v, 1)
        expected = np.sqrt(2) * (point["T_xx"] + point["T_zz"])
        assert abs(point["fixed_chsh"] - expected) < TOL


# ------------------------------------------------------------------
# Horodecki S_max is computed and kept distinct from S_fixed
# ------------------------------------------------------------------

def test_s_max_and_s_fixed_are_separately_reported_and_distinct(mod):
    point = mod.evaluate_point(0.05, 0.05, 0.05, 0.6, 1)
    assert "s_max" in point and "fixed_chsh" in point
    assert point["s_max"] != point["fixed_chsh"]


# ------------------------------------------------------------------
# Correct Q
# ------------------------------------------------------------------

def test_Q_equals_one_minus_Tzz_over_two_for_werner_output(mod):
    for v in (0.2, 0.5, 0.9):
        point = mod.evaluate_point(0.04, 0.0, 0.06, v, 2)
        assert abs(point["Q"] - (1 - point["T_zz"]) / 2) < TOL


# ------------------------------------------------------------------
# DIQKD uses S_fixed, never S_max
# ------------------------------------------------------------------

def test_diqkd_never_uses_s_max_in_source(mod):
    import inspect
    import re

    source = inspect.getsource(mod)
    forbidden = re.findall(r"diqkd_rate\([^)]*\bs_?max\b[^)]*\)", source, re.IGNORECASE)
    assert forbidden == [], f"diqkd_rate called with an S_max-like argument: {forbidden}"


def test_diqkd_rate_reproduces_fixed_chsh_based_value(mod, audit):
    point = mod.evaluate_point(0.02, 0.02, 0.02, 0.9, 1)
    expected_rate, expected_valid = audit.diqkd_rate(point["Q"], point["fixed_chsh"])
    assert point["diqkd_rate_raw"] == expected_rate
    assert point["chsh_valid"] == expected_valid


# ------------------------------------------------------------------
# Threshold consistency
# ------------------------------------------------------------------

def test_thresholds_are_calculated_not_hardcoded(mod):
    """Sanity: root-finding actually locates a crossing where one is
    known to exist (v=1, class 1, matches the previously established
    pure-|Phi+> DIQKD threshold), and correctly reports None where the
    physical range does not contain a root."""
    diqkd_v1 = mod.find_diqkd_threshold_in_p(1.0, 1)
    assert diqkd_v1 is not None
    assert abs(diqkd_v1 - 0.027375113720513045) < 1e-6

    # At p=0.05 (already beyond the v=1 DIQKD threshold), no positive
    # DIQKD rate should exist for any v in [0,1] -- root-finder must
    # report None, not a fabricated value.
    diqkd_p05 = mod.find_diqkd_threshold_in_v(0.05, 1)
    assert diqkd_p05 is None


def test_concurrence_death_visibility_increases_with_noise(mod):
    """Physical sanity check: more channel noise should require higher
    input-state quality (larger v) to remain entangled -- checked
    directly against the calculated thresholds, not assumed."""
    v_crit_low_noise = mod.find_root_bisection(
        lambda v: mod.unclipped_margin_concurrence(0.01, 0.01, 0.01, v, 1), 0.0, 1.0
    )
    v_crit_high_noise = mod.find_root_bisection(
        lambda v: mod.unclipped_margin_concurrence(0.08, 0.08, 0.08, v, 1), 0.0, 1.0
    )
    assert v_crit_low_noise is not None and v_crit_high_noise is not None
    assert v_crit_high_noise > v_crit_low_noise


# ------------------------------------------------------------------
# No regression / no duplication of canonical channels
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
