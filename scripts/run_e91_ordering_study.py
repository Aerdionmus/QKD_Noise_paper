"""
E91 channel-ordering research study
========================================

Purpose
-------
Generates reproducible, machine-readable research data supporting the
D/P/A channel-ordering investigation of the E91/CHSH scenario, for later
use in paper figures/tables. This script does NOT establish new
mathematics and does NOT redefine any canonical channel -- it reuses the
validated helper functions from scripts/audit_e91_channel_ordering.py
(which itself imports the canonical Kraus operators from
src/qkd_noise/channels.py without duplication) and adds:

    - deterministic CSV output under results/e91_ordering/
    - calculated (not hard-coded) DIQKD zero-crossing, CHSH=2, and
      masking-transition thresholds, via robust root finding
    - an explicit STOP-and-report path if any of the established exact
      identities fails to hold numerically (Task 5 of the packaging
      request) -- this script never patches around a disagreement.

What is proven analytically vs. numerically validated vs. literature-based
-------------------------------------------------------------------------
See docs/CHANNEL_ORDERING_ANALYSIS.md and the E91 research memo for the
full derivations. In summary, and NOT re-derived here:

    PROVEN ANALYTICALLY: the two ordering classes, the translation
    difference, the two-sided Delta T_zz formula, Q=(1-T_zz)/2,
    fixed_chsh=sqrt(2)(T_xx+T_zz), the Horodecki eigenvalue structure,
    and the T_zz>=M_x^2 masking criterion.

    NUMERICALLY VALIDATED (by this script, at each run): that the
    simulated density matrices actually satisfy all of the above to
    within floating-point tolerance, across the requested sweeps, plus
    the specific threshold values located by root-finding.

    LITERATURE-BASED, NOT RE-DERIVED: the DIQKD collective-attack bound
    itself (Acin et al., PRL 98, 230501 (2007); Pironio et al., NJP 11,
    045021 (2009)) is applied to the simulated (Q, S_fixed) pairs as a
    published formula -- this script computes its value, it does not
    re-prove it.

    SCIENTIFIC CORRECTION (post-audit, this revision): the DIQKD rate
    is evaluated on `fixed_chsh` (S_fixed) -- the CHSH value from one
    explicit, fixed measurement configuration, matching the protocol
    structure in the Acin et al. (2007) worked example (Alice needs a
    THIRD setting, disjoint from her two CHSH settings, for key
    generation; Bob's key setting is one of his two CHSH settings) --
    and NEVER on `s_max` (the Horodecki state-optimization bound, which
    is not tied to any fixed, realizable protocol configuration and
    must not be treated as "the observed CHSH value"). An earlier
    version of this study incorrectly passed S_max into the DIQKD
    formula; `s_max` is retained in the output purely as a separate
    mathematical diagnostic and is never an input to `diqkd_rate`
    anywhere in this script -- see the docstrings on
    `fixed_setting_chsh` and `diqkd_rate` in
    scripts/audit_e91_channel_ordering.py for the full derivation of
    why S_fixed, under the corrected protocol-consistent role
    assignment, is numerically unchanged from what this script already
    computed (a consequence of our two-sided scenario's diagonal,
    Alice/Bob-symmetric correlation tensor) -- only which quantity feeds
    the DIQKD formula was wrong, not the formula for S_fixed itself.

    NOT CLAIMED anywhere in this script or its output: a new DIQKD
    security theorem, finite-key security, universal E91-over-BB84
    superiority, universal (regime-independent) ordering sensitivity, or
    a direct computation of Eve's Holevo information for this channel
    model.

Run
---
    python scripts/run_e91_ordering_study.py
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "e91_ordering"

TOL = 1e-12
DETECTION_TOL = 1e-9  # for "is this difference physically meaningful" checks


def _load_audit_module():
    """Load scripts/audit_e91_channel_ordering.py by file path (scripts/
    is not a package), and reuse its functions without duplication."""
    script_path = PROJECT_ROOT / "scripts" / "audit_e91_channel_ordering.py"
    spec = importlib.util.spec_from_file_location("audit_e91_channel_ordering", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()

CLASS_1_ORDERING = AUDIT.CANONICAL_CLASS_1_ORDERING
CLASS_2_ORDERING = AUDIT.CANONICAL_CLASS_2_ORDERING


class ExactIdentityViolation(RuntimeError):
    """Raised when an established exact identity fails to hold
    numerically. This script deliberately stops rather than continuing
    past a disagreement with the established mathematics."""


# ----------------------------------------------------------------------
# Core per-point evaluation, with exact-identity regression checks
# ----------------------------------------------------------------------

def evaluate_and_verify(d: float, q: float, gamma: float, cls: int) -> dict:
    """Evaluate one (d,q,gamma,class) point for the two-sided scenario,
    verifying every established exact identity before returning. Raises
    ExactIdentityViolation (stopping the whole run) if any identity is
    violated beyond TOL.
    """
    ordering_name = CLASS_1_ORDERING if cls == 1 else CLASS_2_ORDERING
    sequence = AUDIT.ORDERINGS[ordering_name]

    rho0 = AUDIT.phi_plus()

    # --- Bell-state physicality (checked once per call; cheap) ---
    trace0 = np.trace(rho0)
    if abs(trace0 - 1.0) > TOL:
        raise ExactIdentityViolation(f"Bell state trace != 1: {trace0!r}")
    if np.max(np.abs(rho0 - rho0.conj().T)) > TOL:
        raise ExactIdentityViolation("Bell state not Hermitian")
    if np.min(np.linalg.eigvalsh(rho0)) < -TOL:
        raise ExactIdentityViolation("Bell state not positive semidefinite")

    rho = AUDIT.apply_ordering(rho0, sequence, d, q, gamma, "two")

    trace = np.trace(rho)
    if abs(trace - 1.0) > 1e-9:
        raise ExactIdentityViolation(
            f"Output trace != 1 at d={d} q={q} gamma={gamma} class={cls}: {trace!r}"
        )
    herm_defect = np.max(np.abs(rho - rho.conj().T))
    if herm_defect > 1e-9:
        raise ExactIdentityViolation(
            f"Output not Hermitian at d={d} q={q} gamma={gamma} class={cls}: defect={herm_defect:.3e}"
        )
    min_eig = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2))
    if min_eig < -1e-9:
        raise ExactIdentityViolation(
            f"Output not positive semidefinite at d={d} q={q} gamma={gamma} class={cls}: "
            f"min_eig={min_eig:.3e}"
        )

    T = AUDIT.correlation_tensor(rho)
    Txx, Tyy, Tzz = T[0, 0], T[1, 1], T[2, 2]

    Mx = AUDIT.predicted_Mx(d, q, gamma)
    Mz = AUDIT.predicted_Mz(d, q, gamma)
    tz = AUDIT.predicted_tz(d, q, gamma, cls)

    # --- Exact identity 1: T matches the closed form exactly ---
    expected_T = np.diag([Mx**2, -(Mx**2), Mz**2 + tz**2])
    if np.max(np.abs(T - expected_T)) > TOL:
        raise ExactIdentityViolation(
            f"T mismatch at d={d} q={q} gamma={gamma} class={cls}: "
            f"observed diag={np.diag(T)} expected diag={np.diag(expected_T)}"
        )

    # --- Exact identity 2: Q = (1-T_zz)/2 ---
    Q_direct = AUDIT.key_basis_qber_z(rho)
    Q_formula = (1 - Tzz) / 2
    if abs(Q_direct - Q_formula) > TOL:
        raise ExactIdentityViolation(
            f"Q identity mismatch at d={d} q={q} gamma={gamma} class={cls}: "
            f"direct={Q_direct:.15f} formula={Q_formula:.15f}"
        )

    # --- Exact identity 3: fixed_chsh = sqrt(2)(Txx+Tzz) ---
    S_fixed_direct = AUDIT.fixed_setting_chsh(rho)
    S_fixed_formula = np.sqrt(2) * (Txx + Tzz)
    if abs(S_fixed_direct - S_fixed_formula) > TOL:
        raise ExactIdentityViolation(
            f"fixed_chsh identity mismatch at d={d} q={q} gamma={gamma} class={cls}: "
            f"direct={S_fixed_direct:.15f} formula={S_fixed_formula:.15f}"
        )

    S_max = AUDIT.horodecki_smax(T)
    # SCIENTIFIC CORRECTION (post-audit): the published DIQKD rate must
    # be evaluated on the CHSH value produced by the protocol's actual,
    # fixed measurement configuration (S_fixed_direct, matching the
    # Acin et al. 2007 worked example -- see the docstring on
    # AUDIT.fixed_setting_chsh for the full protocol-consistency
    # derivation), never on S_max, which is a state-optimization bound
    # that no fixed set of protocol measurements directly realizes.
    # S_max is still computed and reported below, but strictly as a
    # separate mathematical diagnostic.
    rate_raw, chsh_valid = AUDIT.diqkd_rate(Q_direct, S_fixed_direct)
    chsh_sensitive = Tzz >= Mx**2  # per the established masking criterion (uses S_max/T_zz, unrelated to DIQKD)

    return {
        "d": d, "q": q, "gamma": gamma, "ordering_class": cls,
        "T_xx": Txx, "T_yy": Tyy, "T_zz": Tzz,
        "Q": Q_direct, "fixed_chsh": S_fixed_direct, "s_max": S_max,
        "diqkd_rate_raw": rate_raw, "chsh_valid": chsh_valid,
        "chsh_sensitive": chsh_sensitive,
    }


# ----------------------------------------------------------------------
# CSV writer helper
# ----------------------------------------------------------------------

CSV_FIELDS = [
    "d", "q", "gamma", "ordering_class",
    "T_xx", "T_yy", "T_zz",
    "Q", "fixed_chsh", "s_max", "diqkd_rate_raw", "chsh_valid", "chsh_sensitive",
    "regime", "shared_p", "p",
]


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in CSV_FIELDS}
            writer.writerow(out_row)


def make_row(point: dict, shared_p, p_value) -> dict:
    row = dict(point)
    row["regime"] = "unmasked" if point["chsh_sensitive"] else "masked"
    row["shared_p"] = shared_p
    row["p"] = p_value if p_value is not None else ""
    # Format floats deterministically; leave bools/strings as-is.
    for key in ("T_xx", "T_yy", "T_zz", "Q", "fixed_chsh", "s_max"):
        row[key] = f"{row[key]:.15e}"
    row["diqkd_rate_raw"] = "" if row["diqkd_rate_raw"] is None else f"{row['diqkd_rate_raw']:.15e}"
    return row


# ----------------------------------------------------------------------
# Robust root finding (bisection on a monotone-enough scanned series;
# falls back to reporting "no root in interval" rather than guessing)
# ----------------------------------------------------------------------

def find_root_bisection(func, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200):
    """Bisection root finder. Returns None if f(lo) and f(hi) do not
    bracket a sign change (reported explicitly by the caller)."""
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo is None or f_hi is None or f_lo == f_hi == 0:
        return None
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = func(mid)
        if f_mid is None:
            return None
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------
# Part A: shared-p sweep
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
# Part B: independent-parameter sweep
# ----------------------------------------------------------------------

def run_independent_parameter_sweep(failures: list) -> list:
    d_values = [0.01, 0.05, 0.10]
    q_values = [0.00, 0.02, 0.05, 0.10]
    gamma_values = [0.01, 0.05, 0.10]
    rows = []
    for d in d_values:
        for q in q_values:
            for gamma in gamma_values:
                for cls in (1, 2):
                    try:
                        point = evaluate_and_verify(d, q, gamma, cls)
                    except ExactIdentityViolation as exc:
                        failures.append(str(exc))
                        continue
                    rows.append(make_row(point, shared_p=False, p_value=None))
    return rows


# ----------------------------------------------------------------------
# Part C: explicit masking-transition sweep (q=0, d=gamma=p, wide range)
# ----------------------------------------------------------------------

def run_masking_transition_sweep(failures: list):
    p_grid = np.round(np.arange(0.01, 1.00, 0.01), 6)
    rows = []
    for p in p_grid:
        for cls in (1, 2):
            try:
                point = evaluate_and_verify(p, 0.0, p, cls)
            except ExactIdentityViolation as exc:
                failures.append(str(exc))
                continue
            rows.append(make_row(point, shared_p=True, p_value=p))

    # Calculate (not hard-code) the masking transition for EACH class
    # SEPARATELY via the analytic masking predicate (T_zz - M_x^2),
    # root-found by bisection. The two classes have different t_z, so
    # they are not guaranteed to share a single transition point --
    # reporting them jointly would hide that (a class-1-only crossing
    # was previously mistaken for a joint one; this version reports
    # both explicitly and honestly).
    def margin_for_class(p, cls):
        Mx = AUDIT.predicted_Mx(p, 0.0, p)
        Mz = AUDIT.predicted_Mz(p, 0.0, p)
        tz = AUDIT.predicted_tz(p, 0.0, p, cls)
        return (Mz**2 + tz**2) - Mx**2

    transition_p_class1 = find_root_bisection(lambda p: margin_for_class(p, 1), 0.01, 0.99, tol=1e-10)
    transition_p_class2 = find_root_bisection(lambda p: margin_for_class(p, 2), 0.01, 0.99, tol=1e-10)
    return rows, {"class1": transition_p_class1, "class2": transition_p_class2}


# ----------------------------------------------------------------------
# Part D: higher-resolution low-noise sweep around the DIQKD region
# ----------------------------------------------------------------------

def run_high_resolution_low_noise_sweep(failures: list) -> list:
    p_grid = np.round(np.arange(0.0001, 0.0301, 0.0001), 6)
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
# Threshold calculations (Task 4): DIQKD zero crossing, CHSH=2, masking
# ----------------------------------------------------------------------

def calculate_thresholds(failures: list) -> dict:
    def rate_fn(p, cls):
        try:
            point = evaluate_and_verify(p, p, p, cls)
        except ExactIdentityViolation as exc:
            failures.append(str(exc))
            return None
        return point["diqkd_rate_raw"]

    def s_fixed_minus_2(p, cls):
        # Protocol-consistent CHSH=2 threshold: uses fixed_chsh (S_fixed),
        # matching the domain boundary actually enforced by diqkd_rate.
        try:
            point = evaluate_and_verify(p, p, p, cls)
        except ExactIdentityViolation as exc:
            failures.append(str(exc))
            return None
        return point["fixed_chsh"] - 2.0

    def s_max_minus_2(p, cls):
        # Separate diagnostic only: the Horodecki-optimum CHSH=2
        # threshold. NOT the domain boundary used by diqkd_rate.
        try:
            point = evaluate_and_verify(p, p, p, cls)
        except ExactIdentityViolation as exc:
            failures.append(str(exc))
            return None
        return point["s_max"] - 2.0

    results = {}
    for cls in (1, 2):
        # DIQKD rate is only defined (non-None) while S_fixed >= 2,
        # which for this shared-p family occurs below p~0.055 (close to,
        # but not identical to, the S_max-based threshold -- see the two
        # separately reported CHSH=2 thresholds below) -- search strictly
        # inside that valid domain rather than up to an arbitrary p that
        # may already be out of range (which would make bisection see a
        # None endpoint and incorrectly report "no root found").
        rate_root = find_root_bisection(lambda p, c=cls: rate_fn(p, c), 0.0001, 0.05, tol=1e-12)
        chsh_fixed_root = find_root_bisection(lambda p, c=cls: s_fixed_minus_2(p, c), 0.0001, 0.20, tol=1e-12)
        chsh_max_root = find_root_bisection(lambda p, c=cls: s_max_minus_2(p, c), 0.0001, 0.20, tol=1e-12)
        results[f"diqkd_zero_crossing_class{cls}"] = rate_root
        results[f"chsh_fixed_eq_2_threshold_class{cls}"] = chsh_fixed_root
        results[f"chsh_max_eq_2_threshold_class{cls}_(diagnostic_only)"] = chsh_max_root
        if rate_root is None:
            failures.append(f"DIQKD zero crossing not found in scanned interval for class {cls}")
        if chsh_fixed_root is None:
            failures.append(f"CHSH(fixed)=2 threshold not found in scanned interval for class {cls}")
        if chsh_max_root is None:
            failures.append(f"CHSH(max)=2 threshold not found in scanned interval for class {cls}")
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    failures: list = []

    print("E91 channel-ordering research study")
    print("========================================")
    print()

    print("Part A: shared-p sweep (p=0.001..0.100)")
    rows_a = run_shared_p_sweep(failures)
    write_csv(RESULTS_DIR / "shared_p_sweep.csv", rows_a)
    print(f"  wrote {len(rows_a)} rows -> results/e91_ordering/shared_p_sweep.csv")

    print("Part B: independent-parameter sweep")
    rows_b = run_independent_parameter_sweep(failures)
    write_csv(RESULTS_DIR / "independent_parameter_sweep.csv", rows_b)
    n_unmasked = sum(1 for r in rows_b if r["regime"] == "unmasked")
    n_masked = sum(1 for r in rows_b if r["regime"] == "masked")
    print(f"  wrote {len(rows_b)} rows -> results/e91_ordering/independent_parameter_sweep.csv"
          f"  (masked={n_masked}, unmasked={n_unmasked})")

    print("Part C: masking-transition sweep (q=0, d=gamma=p, wide range)")
    rows_c, transition_ps = run_masking_transition_sweep(failures)
    write_csv(RESULTS_DIR / "masking_transition_sweep.csv", rows_c)
    print(f"  wrote {len(rows_c)} rows -> results/e91_ordering/masking_transition_sweep.csv")
    print(f"  calculated masking transition (q=0, d=gamma=p), PER CLASS "
          f"(the two classes have different t_z and are not guaranteed to "
          f"share one transition point):")
    print(f"    Class 1 unmasking threshold: p = {transition_ps['class1']}")
    print(f"    Class 2 unmasking threshold: p = {transition_ps['class2']}")

    print("Part D: high-resolution low-noise sweep (DIQKD operating region)")
    rows_d = run_high_resolution_low_noise_sweep(failures)
    write_csv(RESULTS_DIR / "high_resolution_low_noise_sweep.csv", rows_d)
    print(f"  wrote {len(rows_d)} rows -> results/e91_ordering/high_resolution_low_noise_sweep.csv")

    print()
    print("Threshold calculations (robust root finding, not hard-coded)")
    print("------------------------------------------------------------------")
    thresholds = calculate_thresholds(failures)
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    # Class-1-minus-Class-2 difference table (Task 6.5), derived from the
    # shared-p sweep rows already computed above.
    diff_rows = []
    by_p = {}
    for row in rows_a:
        by_p.setdefault(row["p"], {})[row["ordering_class"]] = row
    for p, classes in sorted(by_p.items()):
        if 1 in classes and 2 in classes:
            c1, c2 = classes[1], classes[2]
            def _f(row, key):
                return float(row[key]) if row[key] != "" else None
            r1 = _f(c1, "diqkd_rate_raw")
            r2 = _f(c2, "diqkd_rate_raw")
            diff_rows.append(
                {
                    "p": p,
                    "delta_Q": _f(c1, "Q") - _f(c2, "Q"),
                    "delta_fixed_chsh": _f(c1, "fixed_chsh") - _f(c2, "fixed_chsh"),
                    "delta_s_max": _f(c1, "s_max") - _f(c2, "s_max"),
                    "delta_diqkd_rate": (r1 - r2) if (r1 is not None and r2 is not None) else "",
                }
            )
    diff_path = RESULTS_DIR / "class_difference_summary.csv"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diff_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "delta_Q", "delta_fixed_chsh", "delta_s_max", "delta_diqkd_rate"])
        writer.writeheader()
        for row in diff_rows:
            formatted = dict(row)
            for key in ("delta_Q", "delta_fixed_chsh", "delta_s_max"):
                formatted[key] = f"{formatted[key]:.15e}"
            if formatted["delta_diqkd_rate"] != "":
                formatted["delta_diqkd_rate"] = f"{formatted['delta_diqkd_rate']:.15e}"
            writer.writerow(formatted)
    print(f"  wrote {len(diff_rows)} rows -> results/e91_ordering/class_difference_summary.csv")

    print()
    if failures:
        print(f"STOPPED: {len(failures)} exact-identity violation(s) or missing root(s) found:")
        for msg in failures[:30]:
            print(f"  - {msg}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print("All exact-identity regression checks passed at TOL =", TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
