"""
Tests for the channel-discrimination study
(scripts/run_channel_discrimination_study.py).

Verifies the derived exact result -- that Class 1 and Class 2 differ,
for ANY input state or ancilla-assisted probe, by exactly the constant
operator (Delta_t_z/2) Z, giving a state- and ancilla-independent
diamond norm |Delta_t_z| = 4 d gamma / 3 -- without assuming this in
advance; each property is checked directly.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOL = 1e-9  # numerical (not purely symbolic) checks throughout this file


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_channel_discrimination_study.py"
    spec = importlib.util.spec_from_file_location("run_channel_discrimination_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ------------------------------------------------------------------
# 1-2. Class-1/Class-2 channel difference and state-independence
# ------------------------------------------------------------------

def test_channel_difference_is_state_independent(mod):
    d, q, gamma = 0.07, 0.13, 0.11
    delta_tz = mod.predicted_delta_tz(d, gamma)
    diffs = []
    for name, rho0 in mod.TEST_STATES.items():
        out1 = mod.apply_class_single_qubit(rho0, 1, d, q, gamma)
        out2 = mod.apply_class_single_qubit(rho0, 2, d, q, gamma)
        diffs.append(out1 - out2)
    reference = diffs[0]
    for diff in diffs[1:]:
        assert np.max(np.abs(diff - reference)) < TOL
    expected = (delta_tz / 2) * mod.PAULI_Z
    assert np.max(np.abs(reference - expected)) < TOL


# ------------------------------------------------------------------
# 3. Exact trace-distance formula
# ------------------------------------------------------------------

def test_trace_norm_equals_delta_tz_for_every_test_state(mod):
    d, q, gamma = 0.02, 0.06, 0.09
    delta_tz = mod.predicted_delta_tz(d, gamma)
    for name, rho0 in mod.TEST_STATES.items():
        out1 = mod.apply_class_single_qubit(rho0, 1, d, q, gamma)
        out2 = mod.apply_class_single_qubit(rho0, 2, d, q, gamma)
        tn = mod.trace_norm(out1 - out2)
        assert abs(tn - abs(delta_tz)) < TOL


# ------------------------------------------------------------------
# 4. q-independence
# ------------------------------------------------------------------

def test_trace_norm_independent_of_q(mod):
    d, gamma = 0.05, 0.08
    rho0 = mod.TEST_STATES["|0>"]
    values = set()
    for q in (0.0, 0.1, 0.3, 0.5, 0.9):
        out1 = mod.apply_class_single_qubit(rho0, 1, d, q, gamma)
        out2 = mod.apply_class_single_qubit(rho0, 2, d, q, gamma)
        values.add(round(mod.trace_norm(out1 - out2), 12))
    assert len(values) == 1


# ------------------------------------------------------------------
# 5. Zero-noise limit
# ------------------------------------------------------------------

def test_zero_noise_limit_channels_are_identical(mod):
    rho0 = mod.TEST_STATES["|+>"]
    out1 = mod.apply_class_single_qubit(rho0, 1, 0.0, 0.3, 0.0)
    out2 = mod.apply_class_single_qubit(rho0, 2, 0.0, 0.3, 0.0)
    assert np.max(np.abs(out1 - out2)) < TOL
    assert mod.predicted_delta_tz(0.0, 0.0) == 0.0


# ------------------------------------------------------------------
# 6. Diamond norm: no ancilla (including entangled probes) helps
# ------------------------------------------------------------------

def test_no_ancilla_exceeds_single_qubit_diamond_norm(mod):
    d, q, gamma = 0.06, 0.04, 0.10
    delta_tz = mod.predicted_delta_tz(d, gamma)
    rng = np.random.default_rng(7)
    max_observed = 0.0
    for ancilla_dim in (1, 2, 3):
        for _ in range(30):
            rho_AB = mod.random_density_matrix(2 * ancilla_dim, rng)
            out1 = mod.apply_class_with_ancilla(rho_AB, 1, d, q, gamma, ancilla_dim)
            out2 = mod.apply_class_with_ancilla(rho_AB, 2, d, q, gamma, ancilla_dim)
            tn = mod.trace_norm(out1 - out2)
            max_observed = max(max_observed, tn)
            assert tn <= abs(delta_tz) + 1e-7
    # And a maximally entangled probe achieves exactly the same value
    # (not more, not less) as any single-qubit input.
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    phi_plus = (ket00 + ket11) / np.sqrt(2)
    rho_bell = np.outer(phi_plus, phi_plus.conj())
    out1 = mod.apply_class_with_ancilla(rho_bell, 1, d, q, gamma, 2)
    out2 = mod.apply_class_with_ancilla(rho_bell, 2, d, q, gamma, 2)
    tn_bell = mod.trace_norm(out1 - out2)
    assert abs(tn_bell - abs(delta_tz)) < TOL


def test_diamond_norm_matches_predicted_formula(mod):
    for d, gamma in [(0.01, 0.02), (0.1, 0.1), (0.3, 0.05)]:
        assert abs(mod.predicted_diamond_norm(d, gamma) - 4 * d * gamma / 3) < TOL


# ------------------------------------------------------------------
# 7. Equal-prior success probability
# ------------------------------------------------------------------

def test_p_success_formula(mod):
    for d, gamma in [(0.01, 0.02), (0.1, 0.1)]:
        diamond = mod.predicted_diamond_norm(d, gamma)
        expected = 0.5 + 0.25 * diamond
        assert abs(mod.predicted_p_succ(d, gamma) - expected) < TOL
        assert abs(mod.predicted_p_succ(d, gamma) - (0.5 + d * gamma / 3)) < TOL


# ------------------------------------------------------------------
# 8. Shared-p quadratic scaling
# ------------------------------------------------------------------

def test_shared_p_quadratic_scaling(mod):
    for p in (0.001, 0.01, 0.05, 0.1):
        assert abs(mod.predicted_diamond_norm(p, p) - 4 * p * p / 3) < TOL
        assert abs((mod.predicted_p_succ(p, p) - 0.5) - p * p / 3) < TOL


# ------------------------------------------------------------------
# 9. Independent parameter points
# ------------------------------------------------------------------

def test_independent_parameter_points(mod):
    failures = []
    rng = np.random.default_rng(99)
    for d in (0.01, 0.05, 0.10):
        for gamma in (0.01, 0.05, 0.10):
            point = mod.evaluate_point(d, 0.05, gamma, failures, rng)
            assert abs(point["diamond_norm"] - 4 * d * gamma / 3) < TOL
            for key in (
                "trace_distance_for_|0>", "trace_distance_for_|1>",
                "trace_distance_for_|+>", "trace_distance_for_|->",
                "trace_distance_for_mixed_state",
            ):
                assert abs(point[key] - point["diamond_norm"]) < TOL
    assert failures == []


# ------------------------------------------------------------------
# Does not redefine canonical channels
# ------------------------------------------------------------------

def test_does_not_redefine_canonical_channels(mod):
    from qkd_noise.channels import (
        amplitude_damping_kraus,
        apply_channel,
        dephasing_kraus,
        depolarizing_kraus,
    )

    assert mod.AUDIT.depolarizing_kraus is depolarizing_kraus
    assert mod.AUDIT.dephasing_kraus is dephasing_kraus
    assert mod.AUDIT.amplitude_damping_kraus is amplitude_damping_kraus
    assert mod.AUDIT.apply_channel is apply_channel
