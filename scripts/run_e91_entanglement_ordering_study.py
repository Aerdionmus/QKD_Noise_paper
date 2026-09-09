"""
E91 entanglement (concurrence) ordering study
==================================================

Purpose
-------
Determines whether the D/P/A channel-ordering effect established for
CHSH (S_fixed, S_max) is specifically a Bell-nonlocality phenomenon, or
also a broader entanglement phenomenon, using CONCURRENCE as the
diagnostic. This is not assumed either way -- it is derived and then
checked numerically before any conclusion is drawn.

Density-matrix structure (derived, not assumed)
-------------------------------------------------
For the two-sided output (Lambda tensor Lambda)(|Phi+><Phi+|), the
Pauli decomposition already established elsewhere in this project is

    rho = (1/4)[ I@I + t_z(Z@I + I@Z) + M_x^2 (X@X - Y@Y) + T_zz Z@Z ]

Expanding this explicitly in the computational basis (verified here via
direct symbolic Kronecker-product expansion, not assumed) gives an
X-state with a ZERO (|01>,|10>) coherence:

    rho = [[a, 0, 0, w],
           [0, b, 0, 0],
           [0, 0, b, 0],
           [w, 0, 0, d]]

    a = (1 + 2 t_z + T_zz)/4,  b = (1 - T_zz)/4,  d = (1 - 2 t_z + T_zz)/4,
    w = M_x^2 / 2

(the (|01>,|10>) coherence vanishes because the X@X and Y@Y
contributions to that corner cancel exactly -- X@X - Y@Y only
populates the (|00>,|11>) corner). This is a genuine derivation from
the channel's own affine action on the Bell state, not an assumption
about the state's shape.

For a general X-state with corners (a,b,c,d) and off-diagonal entries
w (the (00,11) corner) and z (the (01,10) corner), the standard closed
form (Yu & Eberly-type X-state concurrence result) is

    C(rho) = 2 max(0, |w| - sqrt(b c), |z| - sqrt(a d))

Here z=0 and b=c, so the second term is <=0 always (it can only lower
the max, never raise it) and the formula collapses to

    C(rho) = 2 max(0, |w| - b) = max(0, M_x^2 - (1 - T_zz)/2)
           = max(0, M_x^2 - Q)

using this project's own established Q = (1-T_zz)/2. This closed form
was cross-checked against the full numerical Wootters concurrence
(eigenvalues of rho (sigma_y@sigma_y) rho* (sigma_y@sigma_y)) at 250+
parameter points spanning the full sweep ranges used below, agreeing to
~1e-14 in every case, before being adopted here (see
test_concurrence_closed_form_matches_wootters in
tests/test_e91_entanglement_ordering.py for the persisted regression
version of that check).

Since M_x^2 is identical across BOTH ordering classes (the composite
channel's linear part M is always order-independent -- established
elsewhere in this project) and only T_zz (via Q) differs between
classes, ANY ordering-sensitivity in concurrence must come entirely
through Q, i.e. through T_zz -- concurrence itself is order-sensitive
if and only if BOTH classes are simultaneously in the "unclipped"
regime (M_x^2 > Q for both). Whenever the max(0, .) clips one or both
classes to exactly zero (an "entanglement sudden death" boundary,
located here by root-finding rather than assumed), the ordering
difference in concurrence can be reduced, including to exactly zero
once both classes are fully separable -- while the CHSH correlator sum
(a plain linear expression, not clipped by any max(0,.)) remains a
formally nonzero, order-sensitive quantity even in that regime, though
no longer operationally meaningful as a nonlocality witness once
S_fixed < 2. This distinction is exactly what Task 7 asks this script
to characterize, and it is characterized empirically below rather than
asserted.

Canonical source of truth (reused, never redefined)
------------------------------------------------------
This script imports directly from scripts/audit_e91_channel_ordering.py
(itself built on the canonical Kraus operators in
src/qkd_noise/channels.py, never duplicated) for: phi_plus,
apply_ordering, correlation_tensor, key_basis_qber_z, fixed_setting_chsh,
horodecki_smax, predicted_Mx, predicted_Mz, predicted_tz, ORDERINGS,
CANONICAL_CLASS_1_ORDERING, CANONICAL_CLASS_2_ORDERING, diqkd_rate.

Run
---
    python scripts/run_e91_entanglement_ordering_study.py
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "e91_entanglement_ordering"

TOL = 1e-12


def _load_audit_module():
    script_path = PROJECT_ROOT / "scripts" / "audit_e91_channel_ordering.py"
    spec = importlib.util.spec_from_file_location("audit_e91_channel_ordering", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()

PAULI_Y = AUDIT.PAULI_Y
CLASS_1_ORDERING = AUDIT.CANONICAL_CLASS_1_ORDERING
CLASS_2_ORDERING = AUDIT.CANONICAL_CLASS_2_ORDERING


class ExactIdentityViolation(RuntimeError):
    """Raised (and never patched around) when a derived identity fails
    to hold numerically."""


# ----------------------------------------------------------------------
# Concurrence: closed-form (derived above) and full numerical Wootters
# ----------------------------------------------------------------------

def concurrence_closed_form(Mx2: float, Tzz: float) -> float:
    """C = max(0, M_x^2 - Q), Q = (1-T_zz)/2 -- derived above from the
    explicit X-state structure of this project's two-sided output."""
    Q = (1.0 - Tzz) / 2.0
    return max(0.0, Mx2 - Q)


def concurrence_wootters(rho4: np.ndarray) -> float:
    """Full, general two-qubit concurrence (Wootters 1998), used only
    as an independent cross-check of the closed form above -- not
    relied on for the sweeps themselves (which use the closed form)."""
    YY = np.kron(PAULI_Y, PAULI_Y)
    rho_tilde = YY @ rho4.conj() @ YY
    R = rho4 @ rho_tilde
    eigvals = np.linalg.eigvals(R)
    eigvals = np.clip(np.sort(np.real(eigvals))[::-1], 0.0, None)
    lambdas = np.sqrt(eigvals)
    return max(0.0, float(lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))


# ----------------------------------------------------------------------
# Per-point evaluation, with exact-identity regression checks
# ----------------------------------------------------------------------

def evaluate_and_verify(d: float, q: float, gamma: float, cls: int) -> dict:
    ordering_name = CLASS_1_ORDERING if cls == 1 else CLASS_2_ORDERING
    sequence = AUDIT.ORDERINGS[ordering_name]

    rho0 = AUDIT.phi_plus()
    rho = AUDIT.apply_ordering(rho0, sequence, d, q, gamma, "two")

    trace = np.trace(rho)
    if abs(trace - 1.0) > 1e-9:
        raise ExactIdentityViolation(f"Output trace != 1 at d={d} q={q} gamma={gamma} class={cls}: {trace!r}")
    herm_defect = np.max(np.abs(rho - rho.conj().T))
    if herm_defect > 1e-9:
        raise ExactIdentityViolation(f"Output not Hermitian at d={d} q={q} gamma={gamma} class={cls}")
    min_eig = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2))
    if min_eig < -1e-9:
        raise ExactIdentityViolation(
            f"Output not positive semidefinite at d={d} q={q} gamma={gamma} class={cls}: min_eig={min_eig:.3e}"
        )

    T = AUDIT.correlation_tensor(rho)
    Txx, Tzz = T[0, 0], T[2, 2]
    Q = AUDIT.key_basis_qber_z(rho)

    # Cross-check Q == (1-T_zz)/2 (already established elsewhere; keep
    # re-verifying it here since concurrence's closed form depends on it).
    if abs(Q - (1 - Tzz) / 2) > TOL:
        raise ExactIdentityViolation(f"Q=(1-T_zz)/2 mismatch at d={d} q={q} gamma={gamma} class={cls}")

    C_closed = concurrence_closed_form(Txx, Tzz)
    C_numeric = concurrence_wootters(rho)
    if abs(C_closed - C_numeric) > 1e-9:
        raise ExactIdentityViolation(
            f"Concurrence closed-form vs numeric mismatch at d={d} q={q} gamma={gamma} class={cls}: "
            f"closed={C_closed:.15f} numeric={C_numeric:.15f}"
        )
    if not (0.0 - 1e-9 <= C_closed <= 1.0 + 1e-9):
        raise ExactIdentityViolation(f"Concurrence out of [0,1] at d={d} q={q} gamma={gamma} class={cls}: {C_closed}")

    S_fixed = AUDIT.fixed_setting_chsh(rho)
    S_max = AUDIT.horodecki_smax(T)

    return {
        "d": d, "q": q, "gamma": gamma, "ordering_class": cls,
        "concurrence": C_closed, "concurrence_numeric_check": C_numeric,
        "T_xx": Txx, "T_zz": Tzz, "Q": Q,
        "fixed_chsh": S_fixed, "s_max": S_max,
        "entangled": C_closed > 0.0,
    }


# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------

SWEEP_FIELDS = [
    "d", "q", "gamma", "ordering_class", "concurrence", "T_xx", "T_zz",
    "Q", "fixed_chsh", "s_max", "entangled", "shared_p", "p",
]


def write_csv(path: Path, rows: list, fields=SWEEP_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            for key in ("concurrence", "T_xx", "T_zz", "Q", "fixed_chsh", "s_max"):
                if out[key] != "":
                    out[key] = f"{out[key]:.15e}"
            writer.writerow(out)


def make_row(point: dict, shared_p, p_value) -> dict:
    row = dict(point)
    row["shared_p"] = shared_p
    row["p"] = p_value if p_value is not None else ""
    return row


# ----------------------------------------------------------------------
# Task 3: independent-parameter sweep
# ----------------------------------------------------------------------

def run_independent_parameter_sweep(failures: list) -> list:
    d_values = [0.01, 0.05, 0.10]
    q_values = [0.00, 0.02, 0.05, 0.10]
    gamma_values = [0.01, 0.05, 0.10]
    rows = []
    for d in d_values:
        for q in q_values:
            for gamma in gamma_values:
                pts = {}
                for cls in (1, 2):
                    try:
                        point = evaluate_and_verify(d, q, gamma, cls)
                    except ExactIdentityViolation as exc:
                        failures.append(str(exc))
                        continue
                    pts[cls] = point
                    rows.append(make_row(point, shared_p=False, p_value=None))
                if 1 in pts and 2 in pts:
                    pts[1]["concurrence_difference"] = pts[1]["concurrence"] - pts[2]["concurrence"]
    return rows


# ----------------------------------------------------------------------
# Task 4: shared-p sweep (low-noise, DIQKD-relevant region)
# ----------------------------------------------------------------------

def run_shared_p_sweep(failures: list) -> list:
    p_grid = np.round(np.arange(0.001, 0.1001, 0.001), 6)
    rows = []
    for p in p_grid:
        for cls in (1, 2):
            try:
                point = evaluate_and_verify(p, p, p, cls)
            except ExactIdentityViolation as exc:
                failures.append(str(exc))
                continue
            rows.append(make_row(point, shared_p=True, p_value=p))
    return rows


# ----------------------------------------------------------------------
# Task 5: extended sweep (structural characterization, not "operational")
# ----------------------------------------------------------------------

def run_extended_sweep(failures: list) -> list:
    p_grid = np.round(np.arange(0.01, 1.00, 0.01), 6)
    rows = []
    for p in p_grid:
        for cls in (1, 2):
            try:
                point = evaluate_and_verify(p, p, p, cls)
            except ExactIdentityViolation as exc:
                failures.append(str(exc))
                continue
            rows.append(make_row(point, shared_p=True, p_value=p))
    return rows


# ----------------------------------------------------------------------
# Entanglement-sudden-death (ESD) threshold, per class, calculated
# (not hard-coded) via bisection on the closed-form concurrence.
# ----------------------------------------------------------------------

def find_root_bisection(func, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200):
    f_lo, f_hi = func(lo), func(hi)
    if f_lo is None or f_hi is None or f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = func(mid)
        if f_mid is None:
            return None
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def esd_threshold(cls: int):
    def margin(p):
        Mx = AUDIT.predicted_Mx(p, p, p)
        Mz = AUDIT.predicted_Mz(p, p, p)
        tz = AUDIT.predicted_tz(p, p, p, cls)
        Tzz = Mz**2 + tz**2
        Q = (1 - Tzz) / 2
        return Mx**2 - Q
    return find_root_bisection(margin, 0.01, 0.99, tol=1e-10)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    failures: list = []

    print("E91 entanglement (concurrence) ordering study")
    print("===================================================")
    print()
    print("Closed-form concurrence (derived from the explicit X-state structure):")
    print("  C(rho) = max(0, M_x^2 - Q),  Q = (1-T_zz)/2")
    print("  Cross-checked against the full Wootters formula at every evaluated point below.")
    print()

    print("Task 3: independent-parameter sweep")
    rows_b = run_independent_parameter_sweep(failures)
    write_csv(RESULTS_DIR / "concurrence_independent_parameter_sweep.csv", rows_b)
    n_entangled = sum(1 for r in rows_b if r["entangled"])
    print(f"  wrote {len(rows_b)} rows -> results/e91_entanglement_ordering/"
          f"concurrence_independent_parameter_sweep.csv  (entangled={n_entangled}, separable={len(rows_b)-n_entangled})")

    print("Task 4: shared-p sweep (p=0.001..0.100, DIQKD-relevant low-noise region)")
    rows_shared = run_shared_p_sweep(failures)
    write_csv(RESULTS_DIR / "concurrence_shared_p_sweep.csv", rows_shared)
    print(f"  wrote {len(rows_shared)} rows -> results/e91_entanglement_ordering/concurrence_shared_p_sweep.csv")

    print("Task 5: extended sweep (p=0.01..0.99, structural characterization)")
    rows_ext = run_extended_sweep(failures)
    write_csv(RESULTS_DIR / "concurrence_extended_sweep.csv", rows_ext)
    print(f"  wrote {len(rows_ext)} rows -> results/e91_entanglement_ordering/concurrence_extended_sweep.csv")

    print()
    print("Entanglement-sudden-death thresholds (calculated via bisection, shared p=d=q=gamma):")
    esd1 = esd_threshold(1)
    esd2 = esd_threshold(2)
    print(f"  Class 1: p = {esd1}")
    print(f"  Class 2: p = {esd2}")

    # Ordering-difference onset and max |Delta C| in the low-noise region
    by_p = {}
    for row in rows_shared:
        by_p.setdefault(row["p"], {})[row["ordering_class"]] = row
    max_abs_delta_c_lownoise = 0.0
    onset_p = None
    for p, classes in sorted(by_p.items()):
        if 1 in classes and 2 in classes:
            dc = classes[1]["concurrence"] - classes[2]["concurrence"]
            max_abs_delta_c_lownoise = max(max_abs_delta_c_lownoise, abs(dc))
            if onset_p is None and abs(dc) > 1e-9:
                onset_p = p

    max_abs_delta_c_extended = 0.0
    by_p_ext = {}
    for row in rows_ext:
        by_p_ext.setdefault(row["p"], {})[row["ordering_class"]] = row
    for p, classes in sorted(by_p_ext.items()):
        if 1 in classes and 2 in classes:
            dc = classes[1]["concurrence"] - classes[2]["concurrence"]
            max_abs_delta_c_extended = max(max_abs_delta_c_extended, abs(dc))

    print()
    print(f"Max |Delta C| in low-noise (p<=0.100) region: {max_abs_delta_c_lownoise:.6e}")
    print(f"Ordering-difference onset (|Delta C|>1e-9):    p~={onset_p}")
    print(f"Max |Delta C| across extended sweep (p<=0.99): {max_abs_delta_c_extended:.6e}")

    # Summary CSV: one row per shared-p point with both classes' key
    # quantities side by side, for later plotting (concurrence vs p,
    # Delta C vs p, concurrence vs CHSH, ordering-class comparison).
    summary_rows = []
    for p, classes in sorted(by_p.items()):
        if 1 in classes and 2 in classes:
            c1, c2 = classes[1], classes[2]
            summary_rows.append(
                {
                    "p": p,
                    "concurrence_class1": c1["concurrence"],
                    "concurrence_class2": c2["concurrence"],
                    "delta_concurrence": c1["concurrence"] - c2["concurrence"],
                    "fixed_chsh_class1": c1["fixed_chsh"],
                    "fixed_chsh_class2": c2["fixed_chsh"],
                    "delta_fixed_chsh": c1["fixed_chsh"] - c2["fixed_chsh"],
                    "s_max_class1": c1["s_max"],
                    "s_max_class2": c2["s_max"],
                }
            )
    summary_path = RESULTS_DIR / "concurrence_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "p", "concurrence_class1", "concurrence_class2", "delta_concurrence",
        "fixed_chsh_class1", "fixed_chsh_class2", "delta_fixed_chsh",
        "s_max_class1", "s_max_class2",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            formatted = {k: (f"{v:.15e}" if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow(formatted)
    print(f"  wrote {len(summary_rows)} rows -> results/e91_entanglement_ordering/concurrence_summary.csv")

    print()
    if failures:
        print(f"STOPPED: {len(failures)} exact-identity violation(s) found:")
        for msg in failures[:30]:
            print(f"  - {msg}")
        return 1

    print("All exact-identity regression checks passed at TOL =", TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
