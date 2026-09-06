"""
Channel-level commutation audit: N_deph o N_dep  vs  N_dep o N_deph
=====================================================================

Purpose
-------
The previous isolated audit (scripts/audit_order_dp.py) found that the
*BB84 QBER* produced by P o D and D o P agree to numerical precision.
Equal BB84 QBER does not by itself imply the two composite quantum
channels are identical as maps on density matrices -- QBER is a
scalar projection of rho onto two measurement bases, so two different
channels could in principle agree on that particular scalar while
disagreeing elsewhere. This script performs a genuine channel-level
audit: it compares full output density matrices (Task 2/3) and, where
practical, the Qiskit superoperator representations of the two
composite channels (Task 4), independent of any QBER computation.

This script is isolated. It does not import, call, or modify
scripts/audit_order_dp.py or anything under results/ordering_audit/
written by the previous audit.

Canonical source of truth (reused, not redefined)
--------------------------------------------------
    src/qkd_noise/channels.py
        depolarizing_kraus(p)  -> Paper Eq. (3), N_dep
        dephasing_kraus(p)     -> Paper Eq. (4), N_deph
        apply_channel(rho, K)  -> Kraus application (used for the
                                   direct density-matrix computation)

Composition-order convention (matches the previous BB84 ordering audit)
-------------------------------------------------------------------------
    P o D := "first D, then P"  = N_deph(N_dep(rho))
    D o P := "first P, then D"  = N_dep(N_deph(rho))

For the density-matrix computation this is done directly with the
canonical Kraus operators via apply_channel(), exactly like the
previous audit.

For the superoperator computation this repository's existing Qiskit
dependency (see requirements.txt: qiskit==0.45.3) is used via
qiskit.quantum_info.Kraus / SuperOp. Qiskit's Kraus.compose(other)
convention was verified empirically before use (see docstring on
_qiskit_composite_kraus below): `A.compose(B)` applies A first, then
B -- i.e. it matches ordinary left-to-right function composition
notation "do A, then do B", NOT the mathematical operator-composition
notation "B after A" read right to left. This was checked directly
against a manual Kraus application on a non-diagonal test state with
two non-commuting unitary Kraus operators (X then a phase gate) before
being relied upon here, specifically so the superoperator order agrees
with the density-matrix order.

If qiskit is not importable in the execution environment, the
superoperator comparison is skipped and clearly reported as skipped;
it is never fabricated.

Outputs (new files only; nothing existing is overwritten)
-------------------------------------------------------------
    results/ordering_audit/dp_channel_commutation.csv
        p, state, frobenius_difference (density-matrix level)
    results/ordering_audit/dp_channel_commutation_superop.csv
        p, superoperator_difference   (only if qiskit available)

Run
---
    python scripts/audit_channel_commutation_dp.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.channels import (  # noqa: E402
    apply_channel,
    dephasing_kraus,
    depolarizing_kraus,
)

TOL = 1e-12

P_VALUES = [round(0.01 * i, 2) for i in range(1, 11)]  # 0.01 .. 0.10

OUTPUT_DIR = PROJECT_ROOT / "results" / "ordering_audit"
DM_CSV_PATH = OUTPUT_DIR / "dp_channel_commutation.csv"
SUPEROP_CSV_PATH = OUTPUT_DIR / "dp_channel_commutation_superop.csv"


# ============================================================
# Task 2 -- general density-matrix test set
# (BB84 states + Bloch-vector states outside the BB84 basis)
# ============================================================

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
    """Returns {label: rho} for BB84 states plus Bloch-vector states.

    Candidate Bloch vectors from the task, checked for |r|^2 <= 1:
        (0.3, 0.4, 0.5)   -> |r|^2 = 0.09+0.16+0.25 = 0.50   VALID
        (-0.2, 0.6, -0.4) -> |r|^2 = 0.04+0.36+0.16 = 0.56   VALID
        (0.7, -0.1, 0.2)  -> |r|^2 = 0.49+0.01+0.04 = 0.54   VALID
    All three candidates are valid as given; no replacement was needed.
    """
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    ket_minus = (ket0 - ket1) / np.sqrt(2)

    states = {
        "|0><0|": np.outer(ket0, ket0.conj()),
        "|1><1|": np.outer(ket1, ket1.conj()),
        "|+><+|": np.outer(ket_plus, ket_plus.conj()),
        "|-><-|": np.outer(ket_minus, ket_minus.conj()),
        "bloch(0.3,0.4,0.5)": _rho_from_bloch(0.3, 0.4, 0.5),
        "bloch(-0.2,0.6,-0.4)": _rho_from_bloch(-0.2, 0.6, -0.4),
        "bloch(0.7,-0.1,0.2)": _rho_from_bloch(0.7, -0.1, 0.2),
    }
    return states


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


# ============================================================
# Task 3 -- direct density-matrix channel comparison
# (uses ONLY the canonical Kraus operators + apply_channel)
# ============================================================

def rho_PD(rho: np.ndarray, p: float) -> np.ndarray:
    """P o D = 'first D, then P' = N_deph(N_dep(rho))."""
    out = apply_channel(rho, depolarizing_kraus(p))
    out = apply_channel(out, dephasing_kraus(p))
    return out


def rho_DP(rho: np.ndarray, p: float) -> np.ndarray:
    """D o P = 'first P, then D' = N_dep(N_deph(rho))."""
    out = apply_channel(rho, dephasing_kraus(p))
    out = apply_channel(out, depolarizing_kraus(p))
    return out


def frobenius_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro"))


# ============================================================
# Task 4 -- superoperator-level comparison (Qiskit, if available)
#
# Uses the repository's existing qiskit dependency
# (requirements.txt: qiskit==0.45.3) and the SAME canonical Kraus
# lists from qkd_noise.channels -- wrapped in qiskit.quantum_info.Kraus
# purely as a container, not redefining the channels.
# ============================================================

def _try_import_qiskit():
    try:
        from qiskit.quantum_info import Kraus, SuperOp  # noqa: F401
        return Kraus, SuperOp
    except Exception:
        return None, None


def superoperator_comparison(p_values):
    """Returns (rows, status_message). rows is [] if skipped."""
    Kraus, SuperOp = _try_import_qiskit()
    if Kraus is None:
        return [], (
            "SKIPPED: qiskit.quantum_info could not be imported in this "
            "environment. No superoperator comparison was performed; this "
            "is reported rather than fabricated. The direct density-matrix "
            "audit (Task 3) above is unaffected and used the canonical "
            "Kraus operators directly via apply_channel()."
        )

    # Empirically verified convention (see module docstring):
    # qiskit's `A.compose(B)` applies A first, then B -- i.e.
    # "do A, then do B", matching our P o D = "first D, then P" notation
    # via D_channel.compose(P_channel), and D o P = "first P, then D"
    # via P_channel.compose(D_channel).
    rows = []
    for p in p_values:
        kraus_D = Kraus(depolarizing_kraus(p))
        kraus_P = Kraus(dephasing_kraus(p))

        composite_PD = kraus_D.compose(kraus_P)  # first D, then P
        composite_DP = kraus_P.compose(kraus_D)  # first P, then D

        superop_PD = SuperOp(composite_PD).data
        superop_DP = SuperOp(composite_DP).data

        diff = float(np.linalg.norm(superop_PD - superop_DP, ord="fro"))
        rows.append({"p": p, "superoperator_difference": diff})

    return rows, "OK: qiskit.quantum_info.Kraus/SuperOp were used successfully."


# ============================================================
# Main audit
# ============================================================

def main() -> int:
    validation_failures: list[str] = []
    states = build_test_states()

    dm_rows = []
    max_dm_diff = -1.0
    max_dm_combo = None

    for p in P_VALUES:
        for label, rho0 in states.items():
            validate_state(rho0, f"input p={p:.2f} state={label}", validation_failures)

            rpd = rho_PD(rho0, p)
            rdp = rho_DP(rho0, p)

            validate_state(rpd, f"P o D output p={p:.2f} state={label}", validation_failures)
            validate_state(rdp, f"D o P output p={p:.2f} state={label}", validation_failures)

            diff = frobenius_diff(rpd, rdp)
            dm_rows.append({"p": p, "state": label, "frobenius_difference": diff})

            if diff > max_dm_diff:
                max_dm_diff = diff
                max_dm_combo = (p, label)

    superop_rows, superop_status = superoperator_comparison(P_VALUES)
    max_superop_diff = max(
        (row["superoperator_difference"] for row in superop_rows), default=None
    )

    # ------------------------------------------------------------
    # Write outputs (new files only)
    # ------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DM_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "state", "frobenius_difference"])
        writer.writeheader()
        for row in dm_rows:
            writer.writerow(row)

    if superop_rows:
        with open(SUPEROP_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["p", "superoperator_difference"])
            writer.writeheader()
            for row in superop_rows:
                writer.writerow(row)

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("Channel-level commutation audit: N_deph o N_dep vs N_dep o N_deph")
    print("====================================================================")
    print()
    print("Canonical channels reused from qkd_noise.channels (unmodified):")
    print("  N_dep  = depolarizing_kraus(p)   [Paper Eq. (3)]")
    print("  N_deph = dephasing_kraus(p)      [Paper Eq. (4)]")
    print(f"  Test states: {len(states)} ({', '.join(states.keys())})")
    print(f"  p sweep: {P_VALUES}")
    print(f"  TOL = {TOL:g}")
    print()

    print("Task 3: direct density-matrix Frobenius differences")
    print("-----------------------------------------------------")
    print(f"{'p':>5} {'state':>22} {'||rho_PD - rho_DP||_F':>24}")
    for row in dm_rows:
        print(f"{row['p']:5.2f} {row['state']:>22} {row['frobenius_difference']:24.6e}")

    print()
    print(f"Maximum density-matrix Frobenius difference = {max_dm_diff:.6e}")
    print(f"  occurs at p = {max_dm_combo[0]:.2f}, state = {max_dm_combo[1]}")
    dm_equal = max_dm_diff <= TOL
    print(f"  Numerically equal at TOL={TOL:g}? {dm_equal}")

    print()
    print("Task 4: superoperator comparison")
    print("---------------------------------")
    print(superop_status)
    if superop_rows:
        print()
        print(f"{'p':>5} {'||S_PD - S_DP||_F':>20}")
        for row in superop_rows:
            print(f"{row['p']:5.2f} {row['superoperator_difference']:20.6e}")
        print()
        print(f"Maximum superoperator Frobenius difference = {max_superop_diff:.6e}")
        superop_equal = max_superop_diff <= TOL
        print(f"  Numerically equal at TOL={TOL:g}? {superop_equal}")

    print()
    print("Physical validation (Task 2)")
    print("-----------------------------")
    if validation_failures:
        print(f"FAILED: {len(validation_failures)} check(s) failed:")
        for msg in validation_failures:
            print(f"  - {msg}")
    else:
        print("All trace / Hermiticity / positivity checks passed for every input and output state.")

    print()
    print("Outputs")
    print("-------")
    print(f"  {DM_CSV_PATH}")
    if superop_rows:
        print(f"  {SUPEROP_CSV_PATH}")
    else:
        print(f"  {SUPEROP_CSV_PATH} (NOT written -- superoperator comparison skipped)")

    return 1 if validation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
