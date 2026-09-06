"""
Tests for the bit-conditioned BB84 QBER asymmetry (Delta_Q_Z).

Checks the properties derived analytically in
docs/CHANNEL_ORDERING_ANALYSIS.md (Bit-Conditioned QBER Asymmetry
addendum):

    Delta_Q_Z := QBER(|0>) - QBER(|1>) = -t_z

    Class 1 (D before A):  Delta_Q_Z = -p
    Class 2 (D after A):   Delta_Q_Z = -p*(1 - 4p/3)
    Class separation:      Delta_Q_Z(Class 1) - Delta_Q_Z(Class 2) = -4p^2/3

    Standard average BB84 QBER remains identical across all six orderings
    (unaffected by the bit-conditioned asymmetry existing underneath it).

These are numerical regression tests at TOL = 1e-12 on a p sweep, not a
substitute for the symbolic derivation in the docs.
"""

import numpy as np

from qkd_noise.channels import (
    amplitude_damping_kraus,
    apply_channel,
    bb84_average_qber,
    bb84_state_qber,
    bb84_states,
    dephasing_kraus,
    depolarizing_kraus,
    projector,
)

TOL = 1e-12

P_VALUES = [0.02, 0.05, 0.13, 0.2]

_KRAUS = {
    "D": depolarizing_kraus,
    "P": dephasing_kraus,
    "A": amplitude_damping_kraus,
}

ORDERINGS = {
    "A o P o D": ("D", "P", "A"),
    "A o D o P": ("P", "D", "A"),
    "P o A o D": ("D", "A", "P"),
    "P o D o A": ("A", "D", "P"),
    "D o A o P": ("P", "A", "D"),
    "D o P o A": ("A", "P", "D"),
}
CLASS_1 = {"A o P o D", "A o D o P", "P o A o D"}
CLASS_2 = {"P o D o A", "D o A o P", "D o P o A"}


def _apply_sequence(rho, sequence, p):
    out = rho.astype(complex, copy=True)
    for label in sequence:
        out = apply_channel(out, _KRAUS[label](p))
    return out


def _delta_q_z(sequence, p):
    bstates = bb84_states()
    rho0 = _apply_sequence(projector(bstates[0]), sequence, p)
    rho1 = _apply_sequence(projector(bstates[1]), sequence, p)
    q0 = bb84_state_qber(rho0, 0)
    q1 = bb84_state_qber(rho1, 1)
    return q0 - q1


def _average_qber(sequence, p):
    bstates = bb84_states()
    qs = []
    for state_id, psi in bstates.items():
        rho = _apply_sequence(projector(psi), sequence, p)
        qs.append(bb84_state_qber(rho, state_id))
    return float(np.mean(qs))


def test_delta_q_z_matches_class_1_prediction():
    """Delta_Q_Z = -p for every Class-1 ordering (D applied before A)."""
    for name in CLASS_1:
        for p in P_VALUES:
            observed = _delta_q_z(ORDERINGS[name], p)
            assert abs(observed - (-p)) < TOL


def test_delta_q_z_matches_class_2_prediction():
    """Delta_Q_Z = -p*(1 - 4p/3) for every Class-2 ordering (D after A)."""
    for name in CLASS_2:
        for p in P_VALUES:
            observed = _delta_q_z(ORDERINGS[name], p)
            predicted = -p * (1 - 4 * p / 3)
            assert abs(observed - predicted) < TOL


def test_class_separation_is_exactly_minus_4p2_over_3():
    """Delta_Q_Z(Class 1) - Delta_Q_Z(Class 2) = -4p^2/3, for every p."""
    for p in P_VALUES:
        c1 = np.mean([_delta_q_z(ORDERINGS[n], p) for n in CLASS_1])
        c2 = np.mean([_delta_q_z(ORDERINGS[n], p) for n in CLASS_2])
        observed_sep = float(c1 - c2)
        predicted_sep = -4 * p * p / 3
        assert abs(observed_sep - predicted_sep) < TOL


def test_delta_q_z_is_nonzero_for_p_greater_than_zero():
    """Sanity check that the asymmetry is a genuine effect, not a
    trivially-zero quantity: at p > 0, Delta_Q_Z != 0 for every ordering."""
    for name, sequence in ORDERINGS.items():
        for p in P_VALUES:
            observed = _delta_q_z(sequence, p)
            assert abs(observed) > 1e-6


def test_average_qber_unaffected_by_bit_conditioned_asymmetry():
    """The standard symmetric average BB84 QBER is still identical across
    all six orderings, even though Delta_Q_Z distinguishes them -- the
    asymmetry exists underneath the average without perturbing it."""
    for p in P_VALUES:
        values = [_average_qber(ORDERINGS[name], p) for name in ORDERINGS]
        assert max(values) - min(values) < TOL


def test_average_qber_matches_existing_triple_scenario():
    """Cross-check against the repository's existing
    bb84_average_qber(p, 'triple') for the canonical 'A o P o D' ordering."""
    for p in P_VALUES:
        local_value = _average_qber(ORDERINGS["A o P o D"], p)
        existing_value = bb84_average_qber(p, "triple")
        assert abs(local_value - existing_value) < TOL
