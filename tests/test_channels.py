import numpy as np

from qkd_noise.channels import (
    I,
    amplitude_damping_kraus,
    apply_channel,
    bb84_average_qber,
    dephasing_kraus,
    depolarizing_kraus,
)


def assert_trace_preserving(kraus_ops):
    acc = sum(k.conj().T @ k for k in kraus_ops)
    np.testing.assert_allclose(acc, I, atol=1e-12)


def test_depolarizing_is_cptp():
    assert_trace_preserving(depolarizing_kraus(0.05))


def test_dephasing_is_cptp():
    assert_trace_preserving(dephasing_kraus(0.05))


def test_amplitude_damping_is_cptp():
    assert_trace_preserving(amplitude_damping_kraus(0.05))


def test_depolarizing_bb84_qber_matches_channel_definition():
    p = 0.05
    assert abs(bb84_average_qber(p, "single") - 2 * p / 3) < 1e-12


def test_zero_noise_is_identity():
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    out = apply_channel(rho, depolarizing_kraus(0.0))
    np.testing.assert_allclose(out, rho, atol=1e-12)
