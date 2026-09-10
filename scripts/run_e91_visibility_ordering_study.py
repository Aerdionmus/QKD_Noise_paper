"""
E91 Werner-state visibility robustness study
=================================================

Purpose
-------
Final robustness experiment: determines whether the D/P/A channel-
ordering effect (established for the ideal |Phi+> state) survives when
the initial entangled state is imperfect, modeled as a Werner state

    rho_W(v) = v |Phi+><Phi+| + (1-v) I_4/4,   0 <= v <= 1.

Nothing here is assumed in advance -- the correlation-tensor formula,
the concurrence closed form, and every threshold are derived and then
checked numerically, exactly as in the prior studies this one extends.

Exact derivation (verified against the actual channel-simulation
pipeline before being adopted here; see the accompanying tests)
-------------------------------------------------------------------
|Phi+> and I_4/4 both have ZERO local marginals and diagonal
correlation tensors (T=diag(1,-1,1) and T=0 respectively), so

    rho_W(v):  r_A = r_B = 0,   T_W = v * diag(1,-1,1)

Since the two-sided affine transformation law derived in the E91 study
only needs r_A=r_B=0 to collapse to T' = M T M^T + t t^T (no
dependence on the specific nonzero value of T, only that it is
diagonal and the marginals vanish), substituting T_W = v*diag(1,-1,1):

    T'(v) = v * (M diag(1,-1,1) M^T) + t t^T
          = diag(v*M_x^2, -v*M_x^2, v*M_z^2 + t_z^2)

    r_A'(v) = r_B'(v) = t_z  (UNCHANGED by v, since the translation
                               term never multiplies the input T at all)

This was confirmed by direct simulation (apply_ordering on an
explicitly constructed Werner state) to match this closed form to
~1e-15, for both classes, across v in [0,1].

Consequence: Delta T_zz = T_zz(C1) - T_zz(C2) = t_z1^2 - t_z2^2
= 8 d gamma^2 (3-2d)/9 -- IDENTICAL to the pure |Phi+> case, with
NO v-dependence whatsoever, since the v-dependent part (v*M_z^2) is
common to both classes and cancels in the difference. This is the
central finding of Task 3, confirmed both analytically and (below,
per-run) numerically -- it is not assumed.

Concurrence: the Werner-state output remains the same X-state shape
established for |Phi+> (zero (|01>,|10>) coherence persists, confirmed
below), so the same closed form applies with the v-dependent T_xx:

    C(v) = max(0, T_xx(v) - Q(v)),  T_xx(v)=v*M_x^2,  Q(v)=(1-T_zz(v))/2

CHSH: S_fixed(v) = sqrt(2)(T_xx(v)+T_zz(v)), same formula as before
with the v-dependent T entries substituted in.

Because Delta T_zz has no v-dependence, but T_xx(v) and the v-dependent
part of T_zz both shrink linearly with v, DECREASING v shrinks the
"signal" (entanglement, CHSH violation, DIQKD-positive region) while
leaving the ordering DIFFERENCE itself unchanged in absolute terms --
so the ordering effect can only become invisible if the underlying
observable itself is driven to (or through) its own zero/threshold
crossing at that v, not because the difference shrinks. This is
checked directly by locating every threshold below, not asserted.

Canonical source of truth (reused, never redefined)
------------------------------------------------------
Imports phi_plus, apply_ordering, correlation_tensor, partial_trace_a/b,
bloch_vector, key_basis_qber_z, fixed_setting_chsh, horodecki_smax,
diqkd_rate, predicted_Mx/Mz/tz, ORDERINGS, and the canonical Kraus
functions (via scripts/audit_e91_channel_ordering.py, itself built on
src/qkd_noise/channels.py) -- nothing is redefined here.

Run
---
    python scripts/run_e91_visibility_ordering_study.py
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "e91_visibility_ordering"

TOL = 1e-12


def _load_audit_module():
    script_path = PROJECT_ROOT / "scripts" / "audit_e91_channel_ordering.py"
    spec = importlib.util.spec_from_file_location("audit_e91_channel_ordering", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()

CLASS_1_ORDERING = AUDIT.CANONICAL_CLASS_1_ORDERING
CLASS_2_ORDERING = AUDIT.CANONICAL_CLASS_2_ORDERING


class ExactIdentityViolation(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Werner state
# ----------------------------------------------------------------------

def werner_state(v: float) -> np.ndarray:
    """rho_W(v) = v |Phi+><Phi+| + (1-v) I_4/4."""
    phi = AUDIT.phi_plus()
    maximally_mixed = np.eye(4, dtype=complex) / 4
    return v * phi + (1 - v) * maximally_mixed


def concurrence_closed_form(Txx: float, Tzz: float) -> float:
    Q = (1.0 - Tzz) / 2.0
    return max(0.0, Txx - Q)


def concurrence_wootters(rho4: np.ndarray) -> float:
    YY = np.kron(AUDIT.PAULI_Y, AUDIT.PAULI_Y)
    rho_tilde = YY @ rho4.conj() @ YY
    R = rho4 @ rho_tilde
    eigvals = np.linalg.eigvals(R)
    eigvals = np.clip(np.sort(np.real(eigvals))[::-1], 0.0, None)
    lambdas = np.sqrt(eigvals)
    return max(0.0, float(lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))


# ----------------------------------------------------------------------
# Per-point evaluation with exact-identity regression checks
# ----------------------------------------------------------------------

def evaluate_point(d: float, q: float, gamma: float, v: float, cls: int) -> dict:
    ordering_name = CLASS_1_ORDERING if cls == 1 else CLASS_2_ORDERING
    sequence = AUDIT.ORDERINGS[ordering_name]

    rho_w = werner_state(v)
    for check_val in (np.trace(rho_w), ):
        if abs(check_val - 1.0) > TOL:
            raise ExactIdentityViolation(f"Werner state trace != 1 at v={v}: {check_val!r}")
    if np.max(np.abs(rho_w - rho_w.conj().T)) > TOL:
        raise ExactIdentityViolation(f"Werner state not Hermitian at v={v}")
    if np.min(np.linalg.eigvalsh(rho_w)) < -1e-10:
        raise ExactIdentityViolation(f"Werner state not positive semidefinite at v={v}")

    rho = AUDIT.apply_ordering(rho_w, sequence, d, q, gamma, "two")

    trace = np.trace(rho)
    if abs(trace - 1.0) > 1e-9:
        raise ExactIdentityViolation(f"Output trace != 1 at d={d} q={q} gamma={gamma} v={v} class={cls}")
    if np.max(np.abs(rho - rho.conj().T)) > 1e-9:
        raise ExactIdentityViolation(f"Output not Hermitian at d={d} q={q} gamma={gamma} v={v} class={cls}")
    min_eig = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2))
    if min_eig < -1e-9:
        raise ExactIdentityViolation(
            f"Output not positive semidefinite at d={d} q={q} gamma={gamma} v={v} class={cls}: min_eig={min_eig:.3e}"
        )

    T = AUDIT.correlation_tensor(rho)
    Txx, Tyy, Tzz = T[0, 0], T[1, 1], T[2, 2]

    Mx = AUDIT.predicted_Mx(d, q, gamma)
    Mz = AUDIT.predicted_Mz(d, q, gamma)
    tz = AUDIT.predicted_tz(d, q, gamma, cls)

    Txx_pred = v * Mx**2
    Tyy_pred = -v * Mx**2
    Tzz_pred = v * Mz**2 + tz**2
    if max(abs(Txx - Txx_pred), abs(Tyy - Tyy_pred), abs(Tzz - Tzz_pred)) > TOL:
        raise ExactIdentityViolation(
            f"T mismatch at d={d} q={q} gamma={gamma} v={v} class={cls}: "
            f"observed=({Txx:.12f},{Tyy:.12f},{Tzz:.12f}) predicted=({Txx_pred:.12f},{Tyy_pred:.12f},{Tzz_pred:.12f})"
        )

    rA = AUDIT.bloch_vector(AUDIT.partial_trace_b(rho))
    rB = AUDIT.bloch_vector(AUDIT.partial_trace_a(rho))
    if max(np.max(np.abs(rA - [0, 0, tz])), np.max(np.abs(rB - [0, 0, tz]))) > TOL:
        raise ExactIdentityViolation(f"Marginal mismatch at d={d} q={q} gamma={gamma} v={v} class={cls}")

    Q_direct = AUDIT.key_basis_qber_z(rho)
    if abs(Q_direct - (1 - Tzz) / 2) > TOL:
        raise ExactIdentityViolation(f"Q=(1-T_zz)/2 mismatch at d={d} q={q} gamma={gamma} v={v} class={cls}")

    S_fixed_direct = AUDIT.fixed_setting_chsh(rho)
    S_fixed_formula = np.sqrt(2) * (Txx + Tzz)
    if abs(S_fixed_direct - S_fixed_formula) > TOL:
        raise ExactIdentityViolation(f"fixed_chsh identity mismatch at d={d} q={q} gamma={gamma} v={v} class={cls}")

    S_max = AUDIT.horodecki_smax(T)

    C_closed = concurrence_closed_form(Txx, Tzz)
    C_numeric = concurrence_wootters(rho)
    if abs(C_closed - C_numeric) > 1e-9:
        raise ExactIdentityViolation(
            f"Concurrence closed-form vs numeric mismatch at d={d} q={q} gamma={gamma} v={v} class={cls}: "
            f"closed={C_closed:.15f} numeric={C_numeric:.15f}"
        )

    # DIQKD rate: MUST use S_fixed, never S_max.
    rate_raw, chsh_valid = AUDIT.diqkd_rate(Q_direct, S_fixed_direct)

    return {
        "d": d, "q": q, "gamma": gamma, "v": v, "ordering_class": cls,
        "T_xx": Txx, "T_yy": Tyy, "T_zz": Tzz,
        "Q": Q_direct, "fixed_chsh": S_fixed_direct, "s_max": S_max,
        "concurrence": C_closed,
        "diqkd_rate_raw": rate_raw, "chsh_valid": chsh_valid,
        "entangled": C_closed > 0.0,
        "chsh_violating": S_fixed_direct > 2.0,
    }


# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------

POINT_FIELDS = [
    "d", "q", "gamma", "v", "ordering_class",
    "T_xx", "T_yy", "T_zz", "Q", "fixed_chsh", "s_max", "concurrence",
    "diqkd_rate_raw", "chsh_valid", "entangled", "chsh_violating", "shared_p", "p",
]


def write_csv(path: Path, rows: list, fields=POINT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_keys = {"T_xx", "T_yy", "T_zz", "Q", "fixed_chsh", "s_max", "concurrence"} & set(fields)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            for key in numeric_keys:
                if out[key] != "":
                    out[key] = f"{out[key]:.15e}"
            if "diqkd_rate_raw" in out and out["diqkd_rate_raw"] not in ("", None):
                out["diqkd_rate_raw"] = f"{out['diqkd_rate_raw']:.15e}"
            if "threshold_value" in out and out["threshold_value"] not in ("", None):
                out["threshold_value"] = f"{out['threshold_value']:.15e}"
            writer.writerow(out)


def make_row(point: dict, shared_p, p_value) -> dict:
    row = dict(point)
    row["shared_p"] = shared_p
    row["p"] = p_value if p_value is not None else ""
    return row


# ----------------------------------------------------------------------
# Root finding
# ----------------------------------------------------------------------

def find_root_bisection(func, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 200):
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


def unclipped_margin_concurrence(d, q, gamma, v, cls):
    Mx = AUDIT.predicted_Mx(d, q, gamma)
    Mz = AUDIT.predicted_Mz(d, q, gamma)
    tz = AUDIT.predicted_tz(d, q, gamma, cls)
    Txx = v * Mx**2
    Tzz = v * Mz**2 + tz**2
    Q = (1 - Tzz) / 2
    return Txx - Q


def s_fixed_minus_2(d, q, gamma, v, cls):
    Mx = AUDIT.predicted_Mx(d, q, gamma)
    Mz = AUDIT.predicted_Mz(d, q, gamma)
    tz = AUDIT.predicted_tz(d, q, gamma, cls)
    Txx = v * Mx**2
    Tzz = v * Mz**2 + tz**2
    return np.sqrt(2) * (Txx + Tzz) - 2.0


def diqkd_rate_of(d, q, gamma, v, cls):
    Mx = AUDIT.predicted_Mx(d, q, gamma)
    Mz = AUDIT.predicted_Mz(d, q, gamma)
    tz = AUDIT.predicted_tz(d, q, gamma, cls)
    Txx = v * Mx**2
    Tzz = v * Mz**2 + tz**2
    Q = (1 - Tzz) / 2
    S = np.sqrt(2) * (Txx + Tzz)
    rate, valid = AUDIT.diqkd_rate(Q, S)
    return rate if valid else None


def find_diqkd_threshold_in_p(v, cls, p_hi_search=0.5):
    """Find the p at which the DIQKD rate crosses zero, restricting the
    search to the region where S_fixed>=2 (the formula's valid domain)
    -- located first via the CHSH=2 boundary in p, rather than blindly
    bisecting into a region where the rate is undefined (None)."""
    p_chsh_boundary = find_root_bisection(lambda p: s_fixed_minus_2(p, p, p, v, cls), 1e-6, p_hi_search)
    if p_chsh_boundary is None:
        return None  # S_fixed never reaches 2 in this range at all
    rate_at_small_p = diqkd_rate_of(1e-6, 1e-6, 1e-6, v, cls)
    rate_near_boundary = diqkd_rate_of(p_chsh_boundary * 0.999, p_chsh_boundary * 0.999, p_chsh_boundary * 0.999, v, cls)
    if rate_at_small_p is None or rate_near_boundary is None:
        return None
    if rate_at_small_p * rate_near_boundary > 0:
        return None  # no sign change within the valid domain
    return find_root_bisection(
        lambda p: diqkd_rate_of(p, p, p, v, cls), 1e-6, p_chsh_boundary * 0.999
    )


def find_diqkd_threshold_in_v(p, cls):
    """Find the v at which the DIQKD rate crosses zero, restricting the
    search to v where S_fixed>=2 (located first via the CHSH=2 boundary
    in v)."""
    v_chsh_boundary = find_root_bisection(lambda v: s_fixed_minus_2(p, p, p, v, cls), 0.0, 1.0)
    if v_chsh_boundary is None:
        return None  # S_fixed never reaches 2 for any v in [0,1] at this p
    rate_at_v1 = diqkd_rate_of(p, p, p, 1.0, cls)
    rate_near_boundary = diqkd_rate_of(p, p, p, min(v_chsh_boundary * 1.001, 1.0), cls)
    if rate_at_v1 is None or rate_near_boundary is None:
        return None
    if rate_at_v1 * rate_near_boundary > 0:
        return None  # no sign change within the valid domain
    return find_root_bisection(
        lambda v: diqkd_rate_of(p, p, p, v, cls), min(v_chsh_boundary * 1.001, 1.0), 1.0
    )


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------

def run_shared_p_sweep(failures: list) -> list:
    p_grid = np.round(np.arange(0.001, 0.1001, 0.001), 6)
    v_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    rows = []
    for p in p_grid:
        for v in v_grid:
            for cls in (1, 2):
                try:
                    point = evaluate_point(p, p, p, v, cls)
                except ExactIdentityViolation as exc:
                    failures.append(str(exc))
                    continue
                rows.append(make_row(point, shared_p=True, p_value=p))
    return rows


def run_independent_sweep(failures: list) -> list:
    d_values = [0.01, 0.05, 0.10]
    q_values = [0.00, 0.05, 0.10]
    gamma_values = [0.01, 0.05, 0.10]
    v_values = [0.6, 0.8, 1.0]
    rows = []
    for d in d_values:
        for q in q_values:
            for gamma in gamma_values:
                for v in v_values:
                    for cls in (1, 2):
                        try:
                            point = evaluate_point(d, q, gamma, v, cls)
                        except ExactIdentityViolation as exc:
                            failures.append(str(exc))
                            continue
                        rows.append(make_row(point, shared_p=False, p_value=None))
    return rows


def run_phase_diagram(failures: list) -> list:
    p_grid = np.round(np.arange(0.005, 0.155, 0.005), 6)
    v_grid = np.round(np.arange(0.50, 1.005, 0.02), 6)
    rows = []
    for p in p_grid:
        for v in v_grid:
            for cls in (1, 2):
                try:
                    point = evaluate_point(p, p, p, v, cls)
                except ExactIdentityViolation as exc:
                    failures.append(str(exc))
                    continue
                rows.append(make_row(point, shared_p=True, p_value=p))
    return rows


def calculate_thresholds(failures: list) -> list:
    """Task 10: thresholds calculated by root-finding, not hard-coded."""
    rows = []

    # (A) fixed v, threshold in p
    for v in (0.5, 0.7, 0.85, 0.9, 1.0):
        for cls in (1, 2):
            chsh_p = find_root_bisection(lambda p, c=cls, vv=v: s_fixed_minus_2(p, p, p, vv, c), 1e-6, 0.5)
            conc_p = find_root_bisection(
                lambda p, c=cls, vv=v: unclipped_margin_concurrence(p, p, p, vv, c), 1e-6, 0.99
            )
            diqkd_p = find_diqkd_threshold_in_p(v, cls)
            rows.append(
                {
                    "fixed_quantity": "v", "fixed_value": v, "ordering_class": cls,
                    "threshold_type": "CHSH_eq_2_in_p", "threshold_value": chsh_p,
                }
            )
            rows.append(
                {
                    "fixed_quantity": "v", "fixed_value": v, "ordering_class": cls,
                    "threshold_type": "concurrence_eq_0_in_p", "threshold_value": conc_p,
                }
            )
            rows.append(
                {
                    "fixed_quantity": "v", "fixed_value": v, "ordering_class": cls,
                    "threshold_type": "DIQKD_eq_0_in_p", "threshold_value": diqkd_p,
                }
            )

    # (B) fixed p, threshold in v
    for p in (0.01, 0.02, 0.05, 0.1):
        for cls in (1, 2):
            chsh_v = find_root_bisection(lambda v, c=cls, pp=p: s_fixed_minus_2(pp, pp, pp, v, c), 0.0, 1.0)
            conc_v = find_root_bisection(
                lambda v, c=cls, pp=p: unclipped_margin_concurrence(pp, pp, pp, v, c), 0.0, 1.0
            )
            diqkd_v = find_diqkd_threshold_in_v(p, cls)
            rows.append(
                {
                    "fixed_quantity": "p", "fixed_value": p, "ordering_class": cls,
                    "threshold_type": "CHSH_eq_2_in_v", "threshold_value": chsh_v,
                }
            )
            rows.append(
                {
                    "fixed_quantity": "p", "fixed_value": p, "ordering_class": cls,
                    "threshold_type": "concurrence_eq_0_in_v", "threshold_value": conc_v,
                }
            )
            rows.append(
                {
                    "fixed_quantity": "p", "fixed_value": p, "ordering_class": cls,
                    "threshold_type": "DIQKD_eq_0_in_v", "threshold_value": diqkd_v,
                }
            )
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    failures: list = []

    print("E91 Werner-state visibility robustness study")
    print("==================================================")
    print()

    print("Task 8: shared-p sweep x visibility grid")
    rows_shared = run_shared_p_sweep(failures)
    write_csv(RESULTS_DIR / "werner_visibility_shared_p.csv", rows_shared)
    print(f"  wrote {len(rows_shared)} rows -> results/e91_visibility_ordering/werner_visibility_shared_p.csv")

    # Verify Delta T_zz is exactly v-independent, using the actual rows.
    by_pv = {}
    for row in rows_shared:
        by_pv.setdefault((row["p"], row["v"]), {})[row["ordering_class"]] = row
    delta_tzz_values = {}
    for (p, v), classes in by_pv.items():
        if 1 in classes and 2 in classes:
            delta_tzz_values.setdefault(p, set()).add(round(classes[1]["T_zz"] - classes[2]["T_zz"], 12))
    n_v_independent = sum(1 for vals in delta_tzz_values.values() if len(vals) == 1)
    print(f"  Delta_T_zz v-independence check: {n_v_independent}/{len(delta_tzz_values)} p-values show "
          f"IDENTICAL Delta_T_zz across all tested v")
    if n_v_independent != len(delta_tzz_values):
        failures.append("Delta_T_zz depends on v -- contradicts the derived analytic prediction")

    print()
    print("Task 9: independent-parameter sweep")
    rows_indep = run_independent_sweep(failures)
    write_csv(RESULTS_DIR / "werner_visibility_independent.csv", rows_indep)
    print(f"  wrote {len(rows_indep)} rows -> results/e91_visibility_ordering/werner_visibility_independent.csv")

    print()
    print("Task 9: two-parameter (p,v) phase diagram")
    rows_phase = run_phase_diagram(failures)
    write_csv(RESULTS_DIR / "werner_visibility_phase_diagram.csv", rows_phase)
    n_entangled = sum(1 for r in rows_phase if r["entangled"])
    n_chsh_violating = sum(1 for r in rows_phase if r["chsh_violating"])
    n_diqkd_positive = sum(
        1 for r in rows_phase if r["diqkd_rate_raw"] not in (None, "") and r["diqkd_rate_raw"] > 0
    )
    print(f"  wrote {len(rows_phase)} rows -> results/e91_visibility_ordering/werner_visibility_phase_diagram.csv")
    print(f"    entangled points: {n_entangled}/{len(rows_phase)}")
    print(f"    CHSH-violating points: {n_chsh_violating}/{len(rows_phase)}")
    print(f"    DIQKD-positive points: {n_diqkd_positive}/{len(rows_phase)}")

    print()
    print("Task 10: thresholds (calculated via root-finding)")
    threshold_rows = calculate_thresholds(failures)
    write_csv(
        RESULTS_DIR / "werner_visibility_thresholds.csv",
        threshold_rows,
        ["fixed_quantity", "fixed_value", "ordering_class", "threshold_type", "threshold_value"],
    )
    print(f"  wrote {len(threshold_rows)} rows -> results/e91_visibility_ordering/werner_visibility_thresholds.csv")
    for row in threshold_rows:
        if row["fixed_quantity"] == "v" and row["fixed_value"] in (0.85, 1.0):
            print(f"    v={row['fixed_value']} class={row['ordering_class']} {row['threshold_type']}: "
                  f"{row['threshold_value']}")
    for row in threshold_rows:
        if row["fixed_quantity"] == "p" and row["fixed_value"] in (0.01, 0.05):
            print(f"    p={row['fixed_value']} class={row['ordering_class']} {row['threshold_type']}: "
                  f"{row['threshold_value']}")

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
