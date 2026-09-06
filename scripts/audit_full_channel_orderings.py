"""
Full D/P/A channel-ordering audit
==================================

Purpose
-------
`scripts/audit_channel_commutation_dp.py` audits the D/P pair only. This
script extends that to the complete ordering question raised in
docs/CHANNEL_ORDERING_ANALYSIS.md:

    1. Pairwise commutation of all three channel pairs (D/P, D/A, P/A).
    2. The equivalence-class structure of all six triple orderings of
       {D, P, A}.
    3. Whether those ordering differences (if any) survive averaging into
       the BB84 QBER observable.

This script is isolated: it does not import, call, or modify
scripts/audit_channel_commutation_dp.py or anything under
results/ordering_audit/ written by that script. It does not overwrite
dp_channel_commutation.csv or dp_channel_commutation_superop.csv.

Canonical source of truth (reused, not redefined)
--------------------------------------------------
    src/qkd_noise/channels.py
        depolarizing_kraus(p), dephasing_kraus(p), amplitude_damping_kraus(p)
        apply_channel(rho, kraus_ops)
        bb84_states(), projector(psi), bb84_state_qber(rho, state_id)

No Kraus operator, parameter convention, or QBER formula is redefined
here. gamma = p throughout, matching bb84_average_qber's "triple"
scenario and docs/PHASE_2_DECISIONS.md.

Composition-order convention
-----------------------------
    X o Y := "apply Y first, then X"    (matches
    scripts/audit_channel_commutation_dp.py and docs/PHASE_2_DECISIONS.md,
    e.g. "N_AD o N_deph o N_dep" = apply depolarizing, then dephasing,
    then amplitude damping).

For the Qiskit SuperOp cross-check, `A.compose(B)` applies A first, then
B (empirically verified in audit_channel_commutation_dp.py); the same
convention is reused here without re-verifying it, since it is the same
qiskit version and the same empirical fact about Kraus.compose.

Outputs (new files only; nothing existing is overwritten)
-----------------------------------------------------------
    results/ordering_audit/full_channel_ordering_superop.csv
        ordering_a, ordering_b, p, superoperator_difference
    results/ordering_audit/full_channel_ordering_qber.csv
        ordering, p, average_bb84_qber

Run
---
    python scripts/audit_full_channel_orderings.py
"""

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.channels import (  # noqa: E402
    amplitude_damping_kraus,
    apply_channel,
    bb84_state_qber,
    bb84_states,
    dephasing_kraus,
    depolarizing_kraus,
    projector,
)

TOL = 1e-12

P_VALUES = [round(0.01 * i, 2) for i in range(1, 11)] + [0.13]  # 0.01..0.10, 0.13

OUTPUT_DIR = PROJECT_ROOT / "results" / "ordering_audit"
SUPEROP_CSV_PATH = OUTPUT_DIR / "full_channel_ordering_superop.csv"
QBER_CSV_PATH = OUTPUT_DIR / "full_channel_ordering_qber.csv"

# Time-ordered application tuples (X1 applied first, X3 applied last),
# keyed by the "X3 o X2 o X1" label used throughout the project docs.
# gamma = p is passed to amplitude_damping_kraus, matching the repo's
# existing "triple" scenario convention.
ORDERINGS = {
    "A o P o D": ("D", "P", "A"),
    "A o D o P": ("P", "D", "A"),
    "P o A o D": ("D", "A", "P"),
    "P o D o A": ("A", "D", "P"),
    "D o A o P": ("P", "A", "D"),
    "D o P o A": ("A", "P", "D"),
}

# Predicted equivalence classes from docs/CHANNEL_ORDERING_ANALYSIS.md
# Section 5 (D before A in time, vs D after A in time). Used only to
# label the report; the audit itself makes no assumption based on this.
PREDICTED_CLASS_1 = {"A o P o D", "A o D o P", "P o A o D"}
PREDICTED_CLASS_2 = {"P o D o A", "D o A o P", "D o P o A"}


def _kraus_for(label: str, p: float):
    if label == "D":
        return depolarizing_kraus(p)
    if label == "P":
        return dephasing_kraus(p)
    if label == "A":
        return amplitude_damping_kraus(p)
    raise ValueError(f"unknown channel label {label!r}")


def _rho_from_bloch(x: float, y: float, z: float) -> np.ndarray:
    """rho = (I + x X + y Y + z Z) / 2. Caller must ensure x^2+y^2+z^2 <= 1."""
    norm_sq = x * x + y * y + z * z
    if norm_sq > 1.0 + 1e-12:
        raise ValueError(
            f"Bloch vector ({x}, {y}, {z}) has |r|^2={norm_sq:.6f} > 1; invalid state."
        )
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return 0.5 * (I + x * X + y * Y + z * Z)


def build_test_states() -> dict:
    """Same 7-state test set as audit_channel_commutation_dp.py, reused
    (not redefined) for consistency with the existing D/P audit."""
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


def validate_state(rho: np.ndarray, label: str, failures: list) -> None:
    trace = np.trace(rho)
    if abs(trace - 1.0) > 1e-8:
        failures.append(f"[{label}] Trace(rho) = {trace!r}, expected ~1")

    herm_defect = np.max(np.abs(rho - rho.conj().T))
    if herm_defect > 1e-10:
        failures.append(f"[{label}] Hermiticity defect = {herm_defect:.3e}")

    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    min_eig = float(np.min(eigvals))
    if min_eig < -1e-9:
        failures.append(f"[{label}] Min eigenvalue = {min_eig:.3e} (positivity violated)")


def frobenius_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro"))


def composite_output(rho: np.ndarray, sequence, p: float) -> np.ndarray:
    """Apply channels in `sequence` (time order, first element first),
    using ONLY the canonical Kraus operators via apply_channel."""
    out = rho.astype(complex, copy=True)
    for label in sequence:
        out = apply_channel(out, _kraus_for(label, p))
    return out


def _try_import_qiskit():
    try:
        from qiskit.quantum_info import Kraus, SuperOp  # noqa: F401
        return Kraus, SuperOp
    except Exception:
        return None, None


def superoperator_for_ordering(sequence, p: float, Kraus, SuperOp):
    """Build the SuperOp for a time-ordered sequence using
    Kraus.compose (A.compose(B) = 'A first, then B'), reusing the
    canonical Kraus lists directly."""
    composite = Kraus(_kraus_for(sequence[0], p))
    for label in sequence[1:]:
        composite = composite.compose(Kraus(_kraus_for(label, p)))
    return SuperOp(composite).data


def numeric_affine_map(kraus_fn, p, label=""):
    """Independent numerical cross-check of the closed-form affine maps
    in docs/CHANNEL_ORDERING_ANALYSIS.md Section 3. Not used for any
    other computation in this script -- purely a reported diagnostic."""
    def bloch_from_rho(rho):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return np.array(
            [
                float(np.real(np.trace(rho @ X))),
                float(np.real(np.trace(rho @ Y))),
                float(np.real(np.trace(rho @ Z))),
            ]
        )

    rho0 = _rho_from_bloch(0.0, 0.0, 0.0)
    t = bloch_from_rho(apply_channel(rho0, kraus_fn(p)))
    cols = []
    for e in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        rho_e = _rho_from_bloch(*e)
        out = bloch_from_rho(apply_channel(rho_e, kraus_fn(p)))
        cols.append(out - t)
    M = np.array(cols).T
    return M, t


def main() -> int:
    validation_failures: list[str] = []
    states = build_test_states()
    ordering_names = list(ORDERINGS.keys())

    # ------------------------------------------------------------
    # Section A: pairwise commutation (D/P, D/A, P/A), density-matrix level
    # ------------------------------------------------------------
    pair_defs = {
        "D/P": ("D", "P"),
        "D/A": ("D", "A"),
        "P/A": ("P", "A"),
    }
    pair_results = {}
    for pair_name, (x, y) in pair_defs.items():
        max_diff = -1.0
        max_combo = None
        for p in P_VALUES:
            for label, rho0 in states.items():
                validate_state(rho0, f"input p={p:.2f} state={label}", validation_failures)
                out_xy = composite_output(rho0, (y, x), p)  # x o y: apply y first
                out_yx = composite_output(rho0, (x, y), p)  # y o x: apply x first
                validate_state(out_xy, f"{x} o {y} output p={p:.2f} state={label}", validation_failures)
                validate_state(out_yx, f"{y} o {x} output p={p:.2f} state={label}", validation_failures)
                diff = frobenius_diff(out_xy, out_yx)
                if diff > max_diff:
                    max_diff = diff
                    max_combo = (p, label)
        pair_results[pair_name] = (max_diff, max_combo)

    # ------------------------------------------------------------
    # Section B: all six triple orderings, density-matrix level
    # ------------------------------------------------------------
    dm_rows = []  # for equivalence-class report (not written to CSV; see Section C for CSV)
    ordering_outputs = {name: {} for name in ordering_names}
    for name, sequence in ORDERINGS.items():
        for p in P_VALUES:
            for label, rho0 in states.items():
                out = composite_output(rho0, sequence, p)
                validate_state(out, f"{name} output p={p:.2f} state={label}", validation_failures)
                ordering_outputs[name][(p, label)] = out

    pairwise_class_diffs = {}
    for name_a, name_b in itertools.combinations(ordering_names, 2):
        max_diff = -1.0
        for p in P_VALUES:
            for label in states:
                diff = frobenius_diff(
                    ordering_outputs[name_a][(p, label)],
                    ordering_outputs[name_b][(p, label)],
                )
                if diff > max_diff:
                    max_diff = diff
        pairwise_class_diffs[(name_a, name_b)] = max_diff

    # Derive numerical equivalence classes via union-find on TOL
    parent = {name: name for name in ordering_names}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b), diff in pairwise_class_diffs.items():
        if diff <= TOL:
            union(a, b)

    classes: dict = {}
    for name in ordering_names:
        root = find(name)
        classes.setdefault(root, []).append(name)
    numerical_classes = list(classes.values())

    # ------------------------------------------------------------
    # Section C: SuperOp cross-check (skipped and reported if unavailable)
    # ------------------------------------------------------------
    Kraus, SuperOp = _try_import_qiskit()
    superop_rows = []
    max_superop_diff = None
    if Kraus is not None:
        for name_a, name_b in itertools.combinations(ordering_names, 2):
            for p in P_VALUES:
                sop_a = superoperator_for_ordering(ORDERINGS[name_a], p, Kraus, SuperOp)
                sop_b = superoperator_for_ordering(ORDERINGS[name_b], p, Kraus, SuperOp)
                diff = float(np.linalg.norm(sop_a - sop_b, ord="fro"))
                superop_rows.append(
                    {
                        "ordering_a": name_a,
                        "ordering_b": name_b,
                        "p": p,
                        "superoperator_difference": diff,
                    }
                )
        max_superop_diff = max(r["superoperator_difference"] for r in superop_rows)
        superop_status = "OK: qiskit.quantum_info.Kraus/SuperOp were used successfully."
    else:
        superop_status = (
            "SKIPPED: qiskit.quantum_info could not be imported in this "
            "environment. No superoperator comparison was performed; this "
            "is reported rather than fabricated. The direct density-matrix "
            "audit above is unaffected."
        )

    # ------------------------------------------------------------
    # Section D: BB84 average QBER by ordering
    # ------------------------------------------------------------
    bstates = bb84_states()
    qber_rows = []
    qber_by_ordering_p = {name: {} for name in ordering_names}
    for name, sequence in ORDERINGS.items():
        for p in P_VALUES:
            qs = []
            for state_id, psi in bstates.items():
                rho = composite_output(projector(psi), sequence, p)
                qs.append(bb84_state_qber(rho, state_id))
            avg = float(np.mean(qs))
            qber_rows.append({"ordering": name, "p": p, "average_bb84_qber": avg})
            qber_by_ordering_p[name][p] = avg

    max_qber_spread = max(
        max(qber_by_ordering_p[name][p] for name in ordering_names)
        - min(qber_by_ordering_p[name][p] for name in ordering_names)
        for p in P_VALUES
    )

    # ------------------------------------------------------------
    # Write outputs (new files only)
    # ------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if superop_rows:
        with open(SUPEROP_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ordering_a", "ordering_b", "p", "superoperator_difference"]
            )
            writer.writeheader()
            for row in superop_rows:
                writer.writerow(row)

    with open(QBER_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ordering", "p", "average_bb84_qber"])
        writer.writeheader()
        for row in qber_rows:
            writer.writerow(row)

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("Full D/P/A channel-ordering audit")
    print("==================================")
    print()
    print("Canonical channels reused from qkd_noise.channels (unmodified):")
    print("  D = depolarizing_kraus(p)        [Paper Eq. (3)]")
    print("  P = dephasing_kraus(p)           [Paper Eq. (4)]")
    print("  A = amplitude_damping_kraus(p)   [Paper Eq. (6), gamma = p]")
    print(f"  Test states: {len(states)} ({', '.join(states.keys())})")
    print(f"  p sweep: {P_VALUES}")
    print(f"  TOL = {TOL:g}")
    print()

    print("Section A: pairwise commutation (max Frobenius diff over all p, states)")
    print("-------------------------------------------------------------------------")
    for pair_name, (max_diff, max_combo) in pair_results.items():
        commute = "COMMUTE" if max_diff <= TOL else "DO NOT COMMUTE"
        print(f"  {pair_name}: max diff = {max_diff:.6e}  ({commute} at TOL={TOL:g})"
              f"  [worst case: p={max_combo[0]:.2f}, state={max_combo[1]}]")
    print()

    print("Section B: triple-ordering pairwise Frobenius differences (max over p, states)")
    print("---------------------------------------------------------------------------------")
    for (a, b), diff in pairwise_class_diffs.items():
        equal = "SAME" if diff <= TOL else "DIFFERENT"
        print(f"  {a:12s} vs {b:12s}: {diff:.6e}  ({equal})")
    print()
    print("Numerical equivalence classes (density-matrix level, TOL=1e-12):")
    for i, cls in enumerate(numerical_classes, start=1):
        print(f"  Class {i}: {sorted(cls)}")
    predicted_ok = {frozenset(c) for c in numerical_classes} == {
        frozenset(PREDICTED_CLASS_1),
        frozenset(PREDICTED_CLASS_2),
    }
    print(f"  Matches analytical prediction in docs/CHANNEL_ORDERING_ANALYSIS.md? {predicted_ok}")
    print()

    print("Section C: SuperOp cross-check")
    print("-------------------------------")
    print(superop_status)
    if superop_rows:
        print(f"  Maximum superoperator Frobenius difference across all ordering pairs = {max_superop_diff:.6e}")
    print()

    print("Section D: average BB84 QBER by ordering")
    print("-------------------------------------------")
    print(f"{'ordering':>12} " + " ".join(f"p={p:<9.2f}" for p in P_VALUES))
    for name in ordering_names:
        vals = " ".join(f"{qber_by_ordering_p[name][p]:<11.8f}" for p in P_VALUES)
        print(f"{name:>12} {vals}")
    print()
    print(f"Maximum spread in average BB84 QBER across all six orderings (any p) = {max_qber_spread:.3e}")
    qber_identical = max_qber_spread <= TOL
    print(f"  Identical across all orderings at TOL={TOL:g}? {qber_identical}")
    print()

    print("Physical validation")
    print("--------------------")
    if validation_failures:
        print(f"FAILED: {len(validation_failures)} check(s) failed:")
        for msg in validation_failures[:20]:
            print(f"  - {msg}")
        if len(validation_failures) > 20:
            print(f"  ... and {len(validation_failures) - 20} more")
    else:
        print("All trace / Hermiticity / positivity checks passed for every input and output state.")
    print()

    print("Outputs")
    print("-------")
    if superop_rows:
        print(f"  {SUPEROP_CSV_PATH}")
    else:
        print(f"  {SUPEROP_CSV_PATH} (NOT written -- superoperator comparison skipped)")
    print(f"  {QBER_CSV_PATH}")

    return 1 if validation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
