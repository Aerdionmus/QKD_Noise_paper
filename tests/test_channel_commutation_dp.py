"""
Isolated test for the channel-level commutation audit (Task 8).

This test computes rho_PD and rho_DP directly from the canonical
Kraus operators (qkd_noise.channels.depolarizing_kraus /
dephasing_kraus, via apply_channel) -- it does NOT depend on any
generated CSV file from scripts/audit_channel_commutation_dp.py.

IMPORTANT SCOPE NOTE:
This is a numerical regression/sanity test, not a mathematical
proof. It demonstrates, to floating-point precision (TOL = 1e-12),
that for THIS repository's specific canonical implementations of
N_dep (Paper Eq. 3) and N_deph (Paper Eq. 4), the composite channels
N_deph(N_dep(rho)) and N_dep(N_deph(rho)) produce numerically
indistinguishable output density matrices on a representative set of
one-qubit states. It does not assert, and should not be read as
asserting, a general theorem about arbitrary quantum channels.
"""

import numpy as np

from qkd_noise.channels import apply_channel, dephasing_kraus, depolarizing_kraus

TOL = 1e-12


def _rho_from_bloch(x: float, y: float, z: float) -> np.ndarray:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return 0.5 * (I + x * X + y * Y + z * Z)


def _test_states():
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    ket_minus = (ket0 - ket1) / np.sqrt(2)
    return [
        np.outer(ket0, ket0.conj()),
        np.outer(ket1, ket1.conj()),
        np.outer(ket_plus, ket_plus.conj()),
        np.outer(ket_minus, ket_minus.conj()),
        _rho_from_bloch(0.3, 0.4, 0.5),
        _rho_from_bloch(-0.2, 0.6, -0.4),
        _rho_from_bloch(0.7, -0.1, 0.2),
    ]


def _rho_PD(rho, p):
    out = apply_channel(rho, depolarizing_kraus(p))
    out = apply_channel(out, dephasing_kraus(p))
    return out


def _rho_DP(rho, p):
    out = apply_channel(rho, dephasing_kraus(p))
    out = apply_channel(out, depolarizing_kraus(p))
    return out


def test_dp_channels_numerically_commute_on_density_matrices():
    """Numerical sanity check only -- see module docstring."""
    p_values = [round(0.01 * i, 2) for i in range(1, 11)]
    states = _test_states()

    max_diff = 0.0
    for p in p_values:
        for rho0 in states:
            rpd = _rho_PD(rho0, p)
            rdp = _rho_DP(rho0, p)
            diff = np.linalg.norm(rpd - rdp, ord="fro")
            max_diff = max(max_diff, diff)

    assert max_diff <= TOL, f"max Frobenius difference {max_diff} exceeds TOL={TOL}"


def test_dp_channel_outputs_remain_valid_density_matrices():
    p_values = [0.01, 0.05, 0.10]
    states = _test_states()

    for p in p_values:
        for rho0 in states:
            for rho in (_rho_PD(rho0, p), _rho_DP(rho0, p)):
                assert abs(np.trace(rho) - 1.0) < 1e-8
                np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)
                eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
                assert eigvals.min() > -1e-9
