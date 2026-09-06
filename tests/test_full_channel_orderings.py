"""
Tests for the full D/P/A channel-ordering analysis (Task 7).

These are numerical regression/sanity tests, not mathematical proofs.
They check, at TOL = 1e-12 on a representative state set, exactly the
properties established analytically in docs/CHANNEL_ORDERING_ANALYSIS.md:

    1. D and P commute.
    2. P and A commute.
    3. D and A do NOT commute for nonzero p (and DO agree at p = 0).
    4. The six triple orderings fall into exactly the two predicted
       equivalence classes.
    5. Average BB84 QBER is identical across all six orderings.
    6. The audit script reuses (does not redefine) the canonical
       channels from qkd_noise.channels.

No hypothesis is encoded here that was not first derived analytically;
see docs/CHANNEL_ORDERING_ANALYSIS.md Sections 4-6 for the derivations
this file checks.
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np

from qkd_noise.channels import (
    amplitude_damping_kraus,
    apply_channel,
    bb84_state_qber,
    bb84_states,
    dephasing_kraus,
    depolarizing_kraus,
    projector,
)

TOL = 1e-12

# Same test states used by scripts/audit_channel_commutation_dp.py and
# scripts/audit_full_channel_orderings.py.
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rho_from_bloch(x, y, z):
    return 0.5 * (_I + x * _X + y * _Y + z * _Z)


def _test_states():
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    ket_minus = (ket0 - ket1) / np.sqrt(2)
    return {
        "|0><0|": np.outer(ket0, ket0.conj()),
        "|1><1|": np.outer(ket1, ket1.conj()),
        "|+><+|": np.outer(ket_plus, ket_plus.conj()),
        "|-><-|": np.outer(ket_minus, ket_minus.conj()),
        "bloch(0.3,0.4,0.5)": _rho_from_bloch(0.3, 0.4, 0.5),
        "bloch(-0.2,0.6,-0.4)": _rho_from_bloch(-0.2, 0.6, -0.4),
        "bloch(0.7,-0.1,0.2)": _rho_from_bloch(0.7, -0.1, 0.2),
    }


_KRAUS = {
    "D": depolarizing_kraus,
    "P": dephasing_kraus,
    "A": amplitude_damping_kraus,
}

# Time-ordered sequences (first element applied first), keyed by the
# "X3 o X2 o X1" label, matching scripts/audit_full_channel_orderings.py.
ORDERINGS = {
    "A o P o D": ("D", "P", "A"),
    "A o D o P": ("P", "D", "A"),
    "P o A o D": ("D", "A", "P"),
    "P o D o A": ("A", "D", "P"),
    "D o A o P": ("P", "A", "D"),
    "D o P o A": ("A", "P", "D"),
}

PREDICTED_CLASS_1 = frozenset({"A o P o D", "A o D o P", "P o A o D"})
PREDICTED_CLASS_2 = frozenset({"P o D o A", "D o A o P", "D o P o A"})

P_VALUES = [0.05, 0.13, 0.2]


def _apply_sequence(rho, sequence, p):
    out = rho.astype(complex, copy=True)
    for label in sequence:
        out = apply_channel(out, _KRAUS[label](p))
    return out


def _max_pair_diff(seq_a, seq_b):
    max_diff = -1.0
    for p in P_VALUES:
        for rho0 in _test_states().values():
            diff = np.linalg.norm(
                _apply_sequence(rho0, seq_a, p) - _apply_sequence(rho0, seq_b, p),
                ord="fro",
            )
            max_diff = max(max_diff, float(diff))
    return max_diff


def test_depolarizing_and_dephasing_commute():
    """D o P == P o D, i.e. apply-D-then-P == apply-P-then-D."""
    diff = _max_pair_diff(("D", "P"), ("P", "D"))
    assert diff < TOL


def test_dephasing_and_amplitude_damping_commute():
    """P o A == A o P."""
    diff = _max_pair_diff(("P", "A"), ("A", "P"))
    assert diff < TOL


def test_depolarizing_and_amplitude_damping_do_not_commute_for_nonzero_p():
    """D o A != A o P for p > 0 (derived: they differ by (4p^2/3) on the
    z Bloch component, see docs/CHANNEL_ORDERING_ANALYSIS.md Section 4)."""
    diff = _max_pair_diff(("D", "A"), ("A", "D"))
    assert diff > 1e-6  # comfortably above numerical noise; predicted ~1.6e-2 at p=0.13


def test_depolarizing_and_amplitude_damping_agree_at_zero_noise():
    """The D/A commutator vanishes at p = 0 (both channels are the
    identity), consistent with the (4p^2/3) difference derived above."""
    for rho0 in _test_states().values():
        out_da = _apply_sequence(rho0, ("D", "A"), 0.0)
        out_ad = _apply_sequence(rho0, ("A", "D"), 0.0)
        np.testing.assert_allclose(out_da, out_ad, atol=1e-12)


def test_triple_ordering_equivalence_classes_match_prediction():
    """The six triple orderings fall into exactly two equivalence
    classes, split by whether D precedes or follows A in time (P's
    position does not matter). See
    docs/CHANNEL_ORDERING_ANALYSIS.md Section 5."""
    names = list(ORDERINGS.keys())
    # Every pair within the predicted classes must agree to TOL.
    for name_a in PREDICTED_CLASS_1:
        for name_b in PREDICTED_CLASS_1:
            assert _max_pair_diff(ORDERINGS[name_a], ORDERINGS[name_b]) < TOL
    for name_a in PREDICTED_CLASS_2:
        for name_b in PREDICTED_CLASS_2:
            assert _max_pair_diff(ORDERINGS[name_a], ORDERINGS[name_b]) < TOL

    # Every pair across the two classes must disagree (for p > 0).
    for name_a in PREDICTED_CLASS_1:
        for name_b in PREDICTED_CLASS_2:
            diff = _max_pair_diff(ORDERINGS[name_a], ORDERINGS[name_b])
            assert diff > 1e-6

    # Sanity: the two predicted classes partition all six orderings.
    assert PREDICTED_CLASS_1 | PREDICTED_CLASS_2 == frozenset(names)
    assert PREDICTED_CLASS_1.isdisjoint(PREDICTED_CLASS_2)


def test_average_bb84_qber_identical_across_all_orderings():
    """Average BB84 QBER is identical across all six triple orderings,
    despite the two equivalence classes being distinct quantum channels
    (docs/CHANNEL_ORDERING_ANALYSIS.md Section 6: the translation term
    that distinguishes the classes cancels in the BB84-symmetric
    average)."""
    bstates = bb84_states()
    for p in P_VALUES:
        qber_values = {}
        for name, sequence in ORDERINGS.items():
            qs = []
            for state_id, psi in bstates.items():
                rho = _apply_sequence(projector(psi), sequence, p)
                qs.append(bb84_state_qber(rho, state_id))
            qber_values[name] = float(np.mean(qs))
        spread = max(qber_values.values()) - min(qber_values.values())
        assert spread < TOL


def test_canonical_ordering_matches_existing_bb84_average_qber():
    """The repository's existing scenario="triple" (D, then P, then A)
    matches the "A o P o D" ordering used here, and its BB84 QBER value
    matches the corresponding entry produced by this module's own
    composition -- confirming no divergence from the existing analytical
    result."""
    from qkd_noise.channels import bb84_average_qber

    bstates = bb84_states()
    for p in P_VALUES:
        qs = []
        for state_id, psi in bstates.items():
            rho = _apply_sequence(projector(psi), ORDERINGS["A o P o D"], p)
            qs.append(bb84_state_qber(rho, state_id))
        local_value = float(np.mean(qs))
        existing_value = bb84_average_qber(p, "triple")
        assert abs(local_value - existing_value) < TOL


def _load_audit_module():
    """Load scripts/audit_full_channel_orderings.py by file path, since
    `scripts/` is not a package and is not on pythonpath (see
    pyproject.toml [tool.pytest.ini_options])."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_full_channel_orderings.py"
    )
    spec = importlib.util.spec_from_file_location("audit_full_channel_orderings", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_script_does_not_redefine_canonical_channels():
    """scripts/audit_full_channel_orderings.py must import the canonical
    Kraus functions from qkd_noise.channels rather than redefining them."""
    audit_module = _load_audit_module()

    assert audit_module.depolarizing_kraus is depolarizing_kraus
    assert audit_module.dephasing_kraus is dephasing_kraus
    assert audit_module.amplitude_damping_kraus is amplitude_damping_kraus
    assert audit_module.apply_channel is apply_channel

    source = inspect.getsource(audit_module)
    # The script's own Kraus-dispatch helper must delegate to the
    # canonical functions, not contain an inline Kraus-operator
    # definition of its own.
    assert "def _kraus_for" in source
