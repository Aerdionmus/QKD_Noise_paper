"""
Bit-conditioned BB84 QBER asymmetry audit
==========================================

Purpose
-------
Extends the ordering audit in scripts/audit_full_channel_orderings.py with
a narrowly focused check of the *bit-conditioned* QBER asymmetry

    Delta_Q_Z(ordering, p) := QBER(|0>) - QBER(|1>)

derived analytically in docs/CHANNEL_ORDERING_ANALYSIS.md (Bit-Conditioned
QBER Asymmetry addendum). Unlike the symmetric average BB84 QBER, this
per-bit-value quantity is NOT protected by the |0>/|1> averaging
cancellation, and is predicted to differ between the two triple-ordering
equivalence classes by exactly

    Delta_Q_Z(Class 1) - Delta_Q_Z(Class 2) = -4 p^2 / 3      (gamma = p)

This script is intentionally narrow: it does not redo the full pairwise-
commutation or SuperOp audit already covered by
scripts/audit_full_channel_orderings.py, and it does not write any CSV
files (per the requested workflow). It reuses the canonical Kraus
functions from qkd_noise.channels without modification or duplication.

Run
---
    python scripts/audit_bit_conditioned_qber.py
"""

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
    bb84_average_qber,
    bb84_state_qber,
    bb84_states,
    dephasing_kraus,
    depolarizing_kraus,
    projector,
)

TOL = 1e-12

P_VALUES = [round(0.01 * i, 2) for i in range(1, 11)] + [0.13]  # 0.01..0.10, 0.13

# Time-ordered sequences (first element applied first), matching
# scripts/audit_full_channel_orderings.py's ORDERINGS table exactly.
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


def _kraus_for(label: str, p: float):
    if label == "D":
        return depolarizing_kraus(p)
    if label == "P":
        return dephasing_kraus(p)
    if label == "A":
        return amplitude_damping_kraus(p)
    raise ValueError(f"unknown channel label {label!r}")


def composite_output(rho: np.ndarray, sequence, p: float) -> np.ndarray:
    out = rho.astype(complex, copy=True)
    for label in sequence:
        out = apply_channel(out, _kraus_for(label, p))
    return out


def bit_conditioned_qber(sequence, p: float) -> dict:
    """Return QBER(|0>), QBER(|1>), Delta_Q_Z, and the standard average
    BB84 QBER for one ordering at one p, using only
    qkd_noise.channels.bb84_state_qber / bb84_states / projector."""
    bstates = bb84_states()
    rho0 = composite_output(projector(bstates[0]), sequence, p)
    rho1 = composite_output(projector(bstates[1]), sequence, p)
    q0 = bb84_state_qber(rho0, 0)
    q1 = bb84_state_qber(rho1, 1)

    qs_all = [q0, q1]
    for state_id in (2, 3):
        rho = composite_output(projector(bstates[state_id]), sequence, p)
        qs_all.append(bb84_state_qber(rho, state_id))
    avg = float(np.mean(qs_all))

    return {"qber_0": q0, "qber_1": q1, "delta_q_z": q0 - q1, "average": avg}


def main() -> int:
    failures: list[str] = []

    print("Bit-conditioned BB84 QBER asymmetry audit")
    print("==========================================")
    print(f"p sweep: {P_VALUES}")
    print(f"TOL = {TOL:g}")
    print()

    per_ordering = {name: {} for name in ORDERINGS}
    for name, seq in ORDERINGS.items():
        for p in P_VALUES:
            per_ordering[name][p] = bit_conditioned_qber(seq, p)

    # ------------------------------------------------------------
    # Check 1: analytical prediction Delta_Q_Z = -t_z per class
    #   Class 1: Delta_Q_Z = -p
    #   Class 2: Delta_Q_Z = -p*(1 - 4p/3)
    # ------------------------------------------------------------
    print("Delta_Q_Z per ordering vs analytical prediction")
    print("-------------------------------------------------")
    for name in ORDERINGS:
        predicted_class = 1 if name in CLASS_1 else 2
        max_err = -1.0
        for p in P_VALUES:
            observed = per_ordering[name][p]["delta_q_z"]
            if predicted_class == 1:
                predicted = -p
            else:
                predicted = -p * (1 - 4 * p / 3)
            err = abs(observed - predicted)
            max_err = max(max_err, err)
            if err > TOL:
                failures.append(
                    f"{name} p={p:.2f}: Delta_Q_Z={observed:.15f} vs "
                    f"predicted={predicted:.15f} (err={err:.3e})"
                )
        print(f"  {name:12s} (Class {predicted_class}): max |observed-predicted| = {max_err:.3e}")
    print()

    # ------------------------------------------------------------
    # Check 2: class separation is exactly -4p^2/3
    # ------------------------------------------------------------
    print("Class separation: Delta_Q_Z(Class 1) - Delta_Q_Z(Class 2)")
    print("-------------------------------------------------------------")
    max_sep_err = -1.0
    for p in P_VALUES:
        class1_vals = [per_ordering[n][p]["delta_q_z"] for n in CLASS_1]
        class2_vals = [per_ordering[n][p]["delta_q_z"] for n in CLASS_2]
        # within-class agreement first
        c1_spread = max(class1_vals) - min(class1_vals)
        c2_spread = max(class2_vals) - min(class2_vals)
        if c1_spread > TOL or c2_spread > TOL:
            failures.append(
                f"p={p:.2f}: within-class Delta_Q_Z spread too large "
                f"(class1={c1_spread:.3e}, class2={c2_spread:.3e})"
            )
        observed_sep = float(np.mean(class1_vals) - np.mean(class2_vals))
        predicted_sep = -4 * p * p / 3
        err = abs(observed_sep - predicted_sep)
        max_sep_err = max(max_sep_err, err)
        if err > TOL:
            failures.append(
                f"p={p:.2f}: class separation={observed_sep:.15f} vs "
                f"predicted={predicted_sep:.15f} (err={err:.3e})"
            )
    print(f"  max |observed - predicted(-4p^2/3)| over all p = {max_sep_err:.3e}")
    print()

    # ------------------------------------------------------------
    # Check 3: standard average BB84 QBER remains ordering-independent
    # ------------------------------------------------------------
    print("Standard average BB84 QBER: ordering independence")
    print("-----------------------------------------------------")
    max_avg_spread = -1.0
    max_ref_diff = -1.0
    for p in P_VALUES:
        avgs = [per_ordering[name][p]["average"] for name in ORDERINGS]
        spread = max(avgs) - min(avgs)
        max_avg_spread = max(max_avg_spread, spread)
        ref = bb84_average_qber(p, "triple")
        diff = abs(avgs[0] - ref)
        max_ref_diff = max(max_ref_diff, diff)
        if spread > TOL:
            failures.append(f"p={p:.2f}: average BB84 QBER spread across orderings = {spread:.3e}")
    print(f"  max spread across all 6 orderings, all p = {max_avg_spread:.3e}")
    print(f"  max |value - bb84_average_qber(p, 'triple')| = {max_ref_diff:.3e}")
    print()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed:")
        for msg in failures[:20]:
            print(f"  - {msg}")
        return 1

    print("All checks passed at TOL =", TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
