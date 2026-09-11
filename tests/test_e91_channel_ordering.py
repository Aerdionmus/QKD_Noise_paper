"""
Tests for the E91/CHSH channel-ordering validation
(scripts/audit_e91_channel_ordering.py).

Verifies, at TOL = 1e-12 (or a looser but still tight tolerance where
noted), the exact predictions from the E91 research memo:

    - Bell-state physicality (normalization, Hermiticity, trace, positivity)
    - one-sided: T is class-independent; local marginal is class-dependent
    - two-sided: T_zz is class-dependent, matching the exact
      8*d*gamma^2*(3-2d)/9 formula (general d,q,gamma)
    - key-basis QBER Q = (1 - T_zz)/2
    - Horodecki S_max formula
    - fixed-setting CHSH formula
    - shared-p leading-order (small-p) behavior
    - zero-noise limit recovers the ideal Bell state

This file imports the helper functions directly from the script by file
path (scripts/ is not a package, consistent with the project's existing
test_full_channel_orderings.py pattern).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qkd_noise.protocols import e91 as legacy_e91

TOL = 1e-12


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_e91_channel_ordering.py"
    )
    spec = importlib.util.spec_from_file_location("audit_e91_channel_ordering", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ------------------------------------------------------------------
# Bell-state physicality
# ------------------------------------------------------------------

def test_bell_state_normalized_hermitian_positive(mod):
    rho = mod.phi_plus()
    assert abs(np.trace(rho) - 1.0) < TOL
    assert np.max(np.abs(rho - rho.conj().T)) < TOL
    eigvals = np.linalg.eigvalsh(rho)
    assert np.min(eigvals) > -TOL


def test_bell_state_correlation_tensor_and_marginals(mod):
    rho = mod.phi_plus()
    T = mod.correlation_tensor(rho)
    np.testing.assert_allclose(T, np.diag([1.0, -1.0, 1.0]), atol=TOL)
    rA = mod.bloch_vector(mod.partial_trace_b(rho))
    rB = mod.bloch_vector(mod.partial_trace_a(rho))
    np.testing.assert_allclose(rA, [0, 0, 0], atol=TOL)
    np.testing.assert_allclose(rB, [0, 0, 0], atol=TOL)


def test_output_states_remain_physical(mod):
    d, q, gamma = 0.07, 0.03, 0.09
    rho0 = mod.phi_plus()
    for name, seq in mod.ORDERINGS.items():
        for sided in ("one", "two"):
            rho = mod.apply_ordering(rho0, seq, d, q, gamma, sided)
            assert abs(np.trace(rho) - 1.0) < 1e-9
            assert np.max(np.abs(rho - rho.conj().T)) < 1e-9
            eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
            assert np.min(eigvals) > -1e-9


# ------------------------------------------------------------------
# One-sided: class-independent T, class-dependent marginal
# ------------------------------------------------------------------

def test_one_sided_correlation_tensor_is_class_independent(mod):
    d, q, gamma = 0.05, 0.02, 0.08
    rho0 = mod.phi_plus()
    T_diffs = []
    for name, seq in mod.ORDERINGS.items():
        rho = mod.apply_ordering(rho0, seq, d, q, gamma, "one")
        T_diffs.append(mod.correlation_tensor(rho))
    reference = T_diffs[0]
    for T in T_diffs[1:]:
        assert np.max(np.abs(T - reference)) < TOL


def test_one_sided_marginal_is_class_dependent(mod):
    d, q, gamma = 0.05, 0.02, 0.08
    rho0 = mod.phi_plus()
    rho1 = mod.apply_ordering(rho0, mod.ORDERINGS["A o P o D"], d, q, gamma, "one")
    rho2 = mod.apply_ordering(rho0, mod.ORDERINGS["P o D o A"], d, q, gamma, "one")
    rA1 = mod.bloch_vector(mod.partial_trace_b(rho1))
    rA2 = mod.bloch_vector(mod.partial_trace_b(rho2))
    tz1 = mod.predicted_tz(d, q, gamma, 1)
    tz2 = mod.predicted_tz(d, q, gamma, 2)
    np.testing.assert_allclose(rA1, [0, 0, tz1], atol=TOL)
    np.testing.assert_allclose(rA2, [0, 0, tz2], atol=TOL)
    assert abs(tz1 - tz2) > 1e-6  # genuinely different, not a degenerate case


def test_one_sided_horodecki_and_fixed_chsh_class_independent(mod):
    d, q, gamma = 0.05, 0.02, 0.08
    rho0 = mod.phi_plus()
    vals_horodecki = []
    vals_fixed = []
    for name, seq in mod.ORDERINGS.items():
        rho = mod.apply_ordering(rho0, seq, d, q, gamma, "one")
        T = mod.correlation_tensor(rho)
        vals_horodecki.append(mod.horodecki_smax(T))
        vals_fixed.append(mod.fixed_setting_chsh(rho))
    assert max(vals_horodecki) - min(vals_horodecki) < TOL
    assert max(vals_fixed) - min(vals_fixed) < TOL


# ------------------------------------------------------------------
# Two-sided: class-dependent T_zz, exact formula
# ------------------------------------------------------------------

def test_two_sided_Tzz_matches_exact_formula_general_dqgamma(mod):
    for d, q, gamma in [(0.02, 0.0, 0.03), (0.05, 0.04, 0.07), (0.10, 0.10, 0.05)]:
        rho0 = mod.phi_plus()
        rho1 = mod.apply_ordering(rho0, mod.ORDERINGS["A o P o D"], d, q, gamma, "two")
        rho2 = mod.apply_ordering(rho0, mod.ORDERINGS["P o D o A"], d, q, gamma, "two")
        T1 = mod.correlation_tensor(rho1)
        T2 = mod.correlation_tensor(rho2)
        observed_diff = T1[2, 2] - T2[2, 2]
        predicted_diff = mod.predicted_delta_Tzz(d, gamma)
        assert abs(observed_diff - predicted_diff) < TOL


def test_two_sided_T_full_formula(mod):
    d, q, gamma = 0.06, 0.03, 0.09
    rho0 = mod.phi_plus()
    for cls, ordering_name in ((1, "A o P o D"), (2, "P o D o A")):
        rho = mod.apply_ordering(rho0, mod.ORDERINGS[ordering_name], d, q, gamma, "two")
        T = mod.correlation_tensor(rho)
        Mx = mod.predicted_Mx(d, q, gamma)
        Mz = mod.predicted_Mz(d, q, gamma)
        tz = mod.predicted_tz(d, q, gamma, cls)
        expected = np.diag([Mx**2, -Mx**2, Mz**2 + tz**2])
        np.testing.assert_allclose(T, expected, atol=TOL)


def test_two_sided_marginals_equal_both_qubits(mod):
    d, q, gamma = 0.06, 0.03, 0.09
    rho0 = mod.phi_plus()
    rho = mod.apply_ordering(rho0, mod.ORDERINGS["A o P o D"], d, q, gamma, "two")
    rA = mod.bloch_vector(mod.partial_trace_b(rho))
    rB = mod.bloch_vector(mod.partial_trace_a(rho))
    tz = mod.predicted_tz(d, q, gamma, 1)
    np.testing.assert_allclose(rA, [0, 0, tz], atol=TOL)
    np.testing.assert_allclose(rB, [0, 0, tz], atol=TOL)


def test_fixed_setting_chsh_equals_sqrt2_times_Txx_plus_Tzz(mod):
    """Regression test for the closed-form reduction of the fixed-setting
    CHSH expression on the two-sided |Phi+> output:

        fixed_setting_chsh(rho) == sqrt(2) * (T_xx + T_zz)

    This identity was derived and confirmed during the DIQKD threshold
    audit but was not previously pinned down by a dedicated test. It is
    checked here directly against the existing `fixed_setting_chsh` and
    `correlation_tensor` functions -- no channel logic is duplicated."""
    parameter_points = [
        (0.02, 0.00, 0.03),
        (0.05, 0.04, 0.07),
        (0.10, 0.10, 0.05),
        (0.001, 0.001, 0.001),
        (0.30, 0.00, 0.30),
    ]
    rho0 = mod.phi_plus()
    for d, q, gamma in parameter_points:
        for ordering_name in ("A o P o D", "P o D o A"):
            rho = mod.apply_ordering(rho0, mod.ORDERINGS[ordering_name], d, q, gamma, "two")
            T = mod.correlation_tensor(rho)
            S_code = mod.fixed_setting_chsh(rho)
            S_formula = np.sqrt(2) * (T[0, 0] + T[2, 2])
            assert abs(S_code - S_formula) < TOL


def test_legacy_e91_chsh_matches_ordering_study_fixed_setting(mod):
    """The executable legacy CHSH value matches S_fixed up to abs()."""
    rho = mod.apply_ordering(
        mod.phi_plus(),
        mod.ORDERINGS["P o D o A"],
        0.05,
        0.03,
        0.04,
        "two",
    )
    fixed_signed = mod.fixed_setting_chsh(rho)
    legacy_absolute = legacy_e91.chsh_value(rho)
    assert abs(legacy_absolute - abs(fixed_signed)) < TOL


# ------------------------------------------------------------------
# Key-basis QBER
# ------------------------------------------------------------------

def test_key_basis_qber_equals_one_minus_Tzz_over_two(mod):
    for d, q, gamma in [(0.02, 0.0, 0.03), (0.08, 0.05, 0.06)]:
        rho0 = mod.phi_plus()
        for ordering_name in ("A o P o D", "P o D o A"):
            rho = mod.apply_ordering(rho0, mod.ORDERINGS[ordering_name], d, q, gamma, "two")
            T = mod.correlation_tensor(rho)
            Q_direct = mod.key_basis_qber_z(rho)
            Q_formula = (1 - T[2, 2]) / 2
            assert abs(Q_direct - Q_formula) < TOL


# ------------------------------------------------------------------
# Horodecki / fixed-setting CHSH formula sanity (ideal state)
# ------------------------------------------------------------------

def test_horodecki_ideal_bell_state_gives_tsirelson_bound(mod):
    rho = mod.phi_plus()
    T = mod.correlation_tensor(rho)
    S_max = mod.horodecki_smax(T)
    assert abs(S_max - 2 * np.sqrt(2)) < TOL


def test_fixed_setting_chsh_ideal_bell_state_gives_tsirelson_bound(mod):
    rho = mod.phi_plus()
    S = mod.fixed_setting_chsh(rho)
    assert abs(S - 2 * np.sqrt(2)) < TOL


# ------------------------------------------------------------------
# Shared-p leading order and zero-noise limit
# ------------------------------------------------------------------

def test_shared_p_small_p_delta_Tzz_leading_order(mod):
    p = 0.001
    predicted = mod.predicted_delta_Tzz(p, p)
    leading_order = (8.0 / 3.0) * p**3
    # leading order should dominate at small p; relative error small
    assert abs(predicted - leading_order) / leading_order < 0.05


def test_zero_noise_limit_recovers_ideal_bell_state(mod):
    rho0 = mod.phi_plus()
    for sided in ("one", "two"):
        for name, seq in mod.ORDERINGS.items():
            rho = mod.apply_ordering(rho0, seq, 0.0, 0.0, 0.0, sided)
            np.testing.assert_allclose(rho, rho0, atol=1e-10)


def test_diqkd_rate_domain_guard(mod):
    # Below classical bound: must not silently clip, must return invalid.
    rate, valid = mod.diqkd_rate(Q=0.1, S=1.9)
    assert valid is False
    assert rate is None
    # At/above threshold: must return a numeric value.
    rate, valid = mod.diqkd_rate(Q=0.01, S=2.6)
    assert valid is True
    assert rate is not None


def test_masking_condition_matches_direct_Tzz_Mx_comparison(mod):
    for d, q, gamma, cls in [(0.05, 0.0, 0.05, 1), (0.05, 0.1, 0.05, 2), (0.4, 0.0, 0.4, 1)]:
        predicted_unmasked = mod.masking_condition(d, q, gamma, cls)
        Mx = mod.predicted_Mx(d, q, gamma)
        Mz = mod.predicted_Mz(d, q, gamma)
        tz = mod.predicted_tz(d, q, gamma, cls)
        direct_unmasked = (Mz**2 + tz**2) >= Mx**2
        assert predicted_unmasked == direct_unmasked


def test_audit_script_does_not_redefine_canonical_channels(mod):
    from qkd_noise.channels import (
        amplitude_damping_kraus,
        apply_channel,
        dephasing_kraus,
        depolarizing_kraus,
    )

    assert mod.depolarizing_kraus is depolarizing_kraus
    assert mod.dephasing_kraus is dephasing_kraus
    assert mod.amplitude_damping_kraus is amplitude_damping_kraus
    assert mod.apply_channel is apply_channel


# ------------------------------------------------------------------
# Scientific correction: S_fixed vs S_max, protocol-consistent DIQKD
# ------------------------------------------------------------------

def test_s_fixed_and_s_max_are_distinct_quantities(mod):
    """S_fixed (one explicit, protocol-realizable measurement
    configuration) and S_max (the Horodecki state-optimization bound)
    must be computed by different functions and must generally differ
    (S_fixed <= S_max), not be silently treated as interchangeable."""
    d, q, gamma = 0.05, 0.05, 0.05
    rho0 = mod.phi_plus()
    rho = mod.apply_ordering(rho0, mod.ORDERINGS["A o P o D"], d, q, gamma, "two")
    T = mod.correlation_tensor(rho)
    S_max = mod.horodecki_smax(T)
    S_fixed = mod.fixed_setting_chsh(rho)
    # They must be computed by genuinely different functions...
    assert mod.horodecki_smax is not mod.fixed_setting_chsh
    # ...and for this anisotropic (T_xx != T_zz) channel, S_fixed is
    # strictly below the Horodecki optimum -- confirming S_max is not
    # a realizable fixed-setting statistic here.
    assert S_fixed < S_max
    assert abs(S_fixed - S_max) > 1e-6


def test_diqkd_rate_never_called_with_s_max_in_source(mod):
    """Regression guard for the audited scientific correction: neither
    scripts/audit_e91_channel_ordering.py nor
    scripts/run_e91_ordering_study.py may pass S_max (Horodecki) into
    diqkd_rate anywhere. This inspects the actual source text rather
    than just one code path, so a future regression cannot silently
    reintroduce the bug in an untested branch."""
    import inspect
    import re

    audit_source = inspect.getsource(mod)
    # No call of the form diqkd_rate(..., <anything containing "S_max"
    # or "s_max" or "smax" as the second argument>) should exist.
    forbidden = re.findall(r"diqkd_rate\([^)]*\bs_?max\b[^)]*\)", audit_source, re.IGNORECASE)
    assert forbidden == [], f"diqkd_rate called with an S_max-like argument: {forbidden}"

    study_path = Path(__file__).resolve().parents[1] / "scripts" / "run_e91_ordering_study.py"
    study_source = study_path.read_text()
    forbidden_study = re.findall(r"diqkd_rate\([^)]*\bs_?max\b[^)]*\)", study_source, re.IGNORECASE)
    assert forbidden_study == [], f"diqkd_rate called with an S_max-like argument: {forbidden_study}"


def test_diqkd_rate_uses_fixed_chsh_value(mod):
    """Sanity check that evaluating diqkd_rate on the actual S_fixed
    value (not S_max) reproduces the study's reported behavior: using
    S_max where S_fixed should be used would give a systematically
    higher (less conservative) rate estimate whenever S_fixed < S_max."""
    d, q, gamma = 0.02, 0.02, 0.02
    rho0 = mod.phi_plus()
    rho = mod.apply_ordering(rho0, mod.ORDERINGS["A o P o D"], d, q, gamma, "two")
    T = mod.correlation_tensor(rho)
    S_max = mod.horodecki_smax(T)
    S_fixed = mod.fixed_setting_chsh(rho)
    Q = mod.key_basis_qber_z(rho)
    rate_fixed, valid_fixed = mod.diqkd_rate(Q, S_fixed)
    rate_max, valid_max = mod.diqkd_rate(Q, S_max)
    assert valid_fixed and valid_max
    # Using S_max would overstate the rate relative to the
    # protocol-consistent S_fixed -- confirming why the correction
    # matters, not merely that the two numbers differ.
    assert rate_max > rate_fixed


def test_key_basis_measurement_matches_protocol_key_pair(mod):
    """Q must come from the (A0=z, B1=z) key-generating pair -- the
    same pair used in the Acin et al. (2007) worked example -- and must
    equal (1-T_zz)/2 exactly, confirming it is not accidentally reusing
    one of the CHSH-only settings or an unrelated definition."""
    for d, q, gamma in [(0.03, 0.0, 0.04), (0.08, 0.05, 0.02)]:
        rho0 = mod.phi_plus()
        rho = mod.apply_ordering(rho0, mod.ORDERINGS["P o D o A"], d, q, gamma, "two")
        T = mod.correlation_tensor(rho)
        Q_direct = mod.key_basis_qber_z(rho)
        assert abs(Q_direct - (1 - T[2, 2]) / 2) < TOL


def test_run_study_reports_both_s_fixed_and_s_max_separately(mod):
    """The study script's per-point evaluation must report S_max
    ('s_max') and the DIQKD-relevant CHSH value ('fixed_chsh')
    as separate, independently-labeled fields."""
    study_path = Path(__file__).resolve().parents[1] / "scripts" / "run_e91_ordering_study.py"
    spec = importlib.util.spec_from_file_location("run_e91_ordering_study", study_path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)

    point = study.evaluate_and_verify(0.03, 0.02, 0.04, 1)
    assert "s_max" in point
    assert "fixed_chsh" in point
    assert point["s_max"] != point["fixed_chsh"]
    # And the rate actually used must match diqkd_rate(Q, fixed_chsh),
    # not diqkd_rate(Q, s_max).
    expected_rate, expected_valid = study.AUDIT.diqkd_rate(point["Q"], point["fixed_chsh"])
    assert point["diqkd_rate_raw"] == expected_rate
    assert point["chsh_valid"] == expected_valid
