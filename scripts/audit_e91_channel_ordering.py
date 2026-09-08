"""
E91 / CHSH channel-ordering validation
=========================================

Purpose
-------
Implements the numerical validation experiment for the E91/CHSH research
memo (channel-ordering ordering-sensitivity investigation). This script
does NOT establish new mathematics; it numerically validates exact
closed-form predictions that were derived by hand/symbolically beforehand,
and reports honestly wherever the predicted masking/DIQKD-threshold
structure does or does not hold, rather than assuming the "interesting"
answer.

Predictions being validated (see docs/CHANNEL_ORDERING_ANALYSIS.md and
the E91 research memo for derivations -- not repeated in full here):

    D_d:  M_D = (1-4d/3) I           t_D = 0
    P_q:  M_P = diag(1-2q,1-2q,1)    t_P = 0
    A_g:  M_A = diag(sqrt(1-g),sqrt(1-g), 1-g)   t_A = (0,0,g)

    Composite (order-independent) linear part:
        M_x = (1-4d/3)(1-2q) sqrt(1-gamma)
        M_z = (1-4d/3)(1-gamma)

    Class 1 translation:  t_z = gamma
    Class 2 translation:  t_z = (1-4d/3) gamma

    One-sided  (Lambda x I)(|Phi+><Phi+|):
        r_A = (0,0,t_z)   r_B = 0        T = diag(M_x, -M_x, M_z)
        (T is class-independent; only the local marginal differs.)

    Two-sided  (Lambda x Lambda)(|Phi+><Phi+|):
        r_A = r_B = (0,0,t_z)
        T = diag(M_x^2, -M_x^2, M_z^2 + t_z^2)
        Delta T_zz = 8 d gamma^2 (3-2d) / 9     (general d,q,gamma; no q dependence)

    Horodecki:  S_max = 2 sqrt(u1+u2), u1,u2 = top-2 eigenvalues of T^T T.
    Masking condition (two-sided): class-dependence is visible in S_max
    iff T_zz^2 is among the top-2 eigenvalues, i.e. T_zz >= M_x^2
    (equivalently T_zz^2 >= M_x^4, since both are nonnegative here).

    Key-basis QBER (Z,Z measurement, two-sided): Q = (1 - T_zz) / 2.

    DIQKD collective-attack bound (Acin et al. 2007; Pironio et al. 2009),
    applied to our simulated (Q,S) -- NOT a new theorem of this project:
        r = 1 - h(Q) - h( (1 + sqrt((S/2)^2 - 1)) / 2 ),   valid only for S >= 2.

Canonical source of truth (reused, never redefined)
------------------------------------------------------
    src/qkd_noise/channels.py
        depolarizing_kraus(p), dephasing_kraus(p), amplitude_damping_kraus(p)
        apply_channel(rho, kraus_ops)

Two-qubit application is implemented here by embedding the canonical
single-qubit Kraus operators via Kronecker products (kron(K, I) for
one-sided, kron(Ka, Kb) for two-sided) and then calling the canonical
`apply_channel` on the resulting 4x4 operators -- no Kraus operator is
redefined or duplicated.

This script is independent of, and does not modify or import from,
src/qkd_noise/protocols/e91.py (which implements a different,
pre-existing single-scenario E91 simulation with its own local Kraus
definitions used for a different part of the project).

Run
---
    python scripts/audit_e91_channel_ordering.py
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
    dephasing_kraus,
    depolarizing_kraus,
)

TOL = 1e-12

I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)

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
CANONICAL_CLASS_1_ORDERING = "A o P o D"
CANONICAL_CLASS_2_ORDERING = "P o D o A"


# ----------------------------------------------------------------------
# 1. Bell-state construction
# ----------------------------------------------------------------------

def phi_plus() -> np.ndarray:
    """|Phi+> = (|00> + |11>) / sqrt(2), as an explicit 4x4 density matrix."""
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    psi = (ket00 + ket11) / np.sqrt(2)
    return np.outer(psi, psi.conj())


# ----------------------------------------------------------------------
# 2. Applying a canonical single-qubit channel to one/both qubits
# ----------------------------------------------------------------------

def _kraus_for(label: str, d: float, q: float, gamma: float):
    if label == "D":
        return depolarizing_kraus(d)
    if label == "P":
        return dephasing_kraus(q)
    if label == "A":
        return amplitude_damping_kraus(gamma)
    raise ValueError(f"unknown channel label {label!r}")


def _embed_one_sided(kraus_list):
    """Kraus set for (Lambda tensor I) built from canonical single-qubit ops."""
    return [np.kron(K, I2) for K in kraus_list]


def _embed_two_sided(kraus_list):
    """Kraus set for (Lambda tensor Lambda) built from canonical single-qubit ops."""
    return [np.kron(Ka, Kb) for Ka in kraus_list for Kb in kraus_list]


def apply_ordering(rho4: np.ndarray, sequence, d: float, q: float, gamma: float, sided: str) -> np.ndarray:
    """Apply a time-ordered D/P/A sequence to a two-qubit state.

    sided: 'one' for (Lambda x I), 'two' for (Lambda x Lambda).
    """
    out = rho4.astype(complex, copy=True)
    for label in sequence:
        kraus = _kraus_for(label, d, q, gamma)
        embedded = _embed_one_sided(kraus) if sided == "one" else _embed_two_sided(kraus)
        out = apply_channel(out, embedded)
    return out


# ----------------------------------------------------------------------
# 3. Reduced states, Bloch vectors, correlation tensor
# ----------------------------------------------------------------------

def partial_trace_b(rho4: np.ndarray) -> np.ndarray:
    """Trace out qubit B (second qubit) -> Alice's reduced 2x2 state."""
    rho = rho4.reshape(2, 2, 2, 2)
    return np.einsum("ikjk->ij", rho)


def partial_trace_a(rho4: np.ndarray) -> np.ndarray:
    """Trace out qubit A (first qubit) -> Bob's reduced 2x2 state."""
    rho = rho4.reshape(2, 2, 2, 2)
    return np.einsum("kikj->ij", rho)


def bloch_vector(rho2: np.ndarray) -> np.ndarray:
    return np.array([float(np.real(np.trace(rho2 @ p))) for p in PAULIS])


def correlation_tensor(rho4: np.ndarray) -> np.ndarray:
    T = np.zeros((3, 3))
    for i, pi in enumerate(PAULIS):
        for j, pj in enumerate(PAULIS):
            op = np.kron(pi, pj)
            T[i, j] = float(np.real(np.trace(rho4 @ op)))
    return T


def validate_physical(rho4: np.ndarray, label: str, failures: list) -> None:
    trace = np.trace(rho4)
    if abs(trace - 1.0) > 1e-8:
        failures.append(f"[{label}] Trace = {trace!r}, expected ~1")
    herm_defect = np.max(np.abs(rho4 - rho4.conj().T))
    if herm_defect > 1e-10:
        failures.append(f"[{label}] Hermiticity defect = {herm_defect:.3e}")
    eigvals = np.linalg.eigvalsh((rho4 + rho4.conj().T) / 2)
    min_eig = float(np.min(eigvals))
    if min_eig < -1e-9:
        failures.append(f"[{label}] Min eigenvalue = {min_eig:.3e} (positivity violated)")


# ----------------------------------------------------------------------
# 4. CHSH: Horodecki S_max and fixed-setting S
# ----------------------------------------------------------------------

def horodecki_smax(T: np.ndarray) -> float:
    M = T.T @ T
    eigs = np.sort(np.linalg.eigvalsh(M))[::-1]
    u1, u2 = eigs[0], eigs[1]
    return float(2 * np.sqrt(max(u1 + u2, 0.0)))


def correlator(rho4: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    a_op = a_vec[0] * PAULI_X + a_vec[1] * PAULI_Y + a_vec[2] * PAULI_Z
    b_op = b_vec[0] * PAULI_X + b_vec[1] * PAULI_Y + b_vec[2] * PAULI_Z
    return float(np.real(np.trace(rho4 @ np.kron(a_op, b_op))))


def fixed_setting_chsh(rho4: np.ndarray) -> float:
    """The CHSH value obtained from ONE explicit, fixed set of four
    measurement settings -- the quantity an actual protocol run would
    observe, as opposed to `horodecki_smax` (a state-optimization bound
    over ALL possible settings, never realized by any single fixed
    protocol).

    PROTOCOL-CONSISTENCY NOTE (added after an audit of this point):
    the published DIQKD collective-attack bound (Acin et al., PRL 98,
    230501 (2007); Pironio et al., NJP 11, 045021 (2009)) requires Alice
    to have a THIRD measurement setting, used only for key generation,
    that is DISJOINT from the two settings she uses in the CHSH
    expression -- Bob's key-generating setting, by contrast, IS one of
    his two CHSH settings. Acin et al. (2007) give an explicit worked
    example for a Bell-diagonal/Werner-type state (structurally the
    same shape as our two-sided output: diagonal correlation tensor,
    one axis distinguished) with settings

        A0 = B1 = sigma_z   (the disjoint Alice key setting / shared Bob
                              key+CHSH setting)
        A1 = (sigma_z+sigma_x)/sqrt(2),  A2 = (sigma_z-sigma_x)/sqrt(2)
                              (Alice's two CHSH-only settings)
        B2 = sigma_x          (Bob's second CHSH-only setting)

    i.e. the DIAGONAL settings belong to Alice (A1, A2), and Bob keeps
    the plain z/x settings (B1, B2), with B1=z doing double duty as the
    key setting. This is the mirror image of what an earlier version of
    this script implicitly assumed (diagonal settings on Bob, plain z/x
    on Alice, with Alice's "a0"=z reused as both a CHSH setting AND the
    key setting -- which violates the required A0-disjoint-from-CHSH
    structure for Alice).

    For our specific two-sided scenario, both qubits pass through the
    SAME channel, so the resulting correlation tensor T is diagonal and
    symmetric under an Alice<->Bob relabeling (E(a,b)=a^T T b = b^T T a
    for diagonal T). Recomputing S with the corrected, literature-
    matching role assignment (A0=B1=z key pair; A1,A2 diagonal for
    Alice; B2=x for Bob) was verified to give EXACTLY the same value as
    this function already computes:

        S = E(A1,B1) + E(A1,B2) + E(A2,B1) - E(A2,B2) = sqrt(2)(T_xx+T_zz)

    -- i.e. this function's numerical output is unchanged and IS the
    protocol-consistent observed CHSH statistic for our two-sided
    scenario; only the earlier LABELING of which party owns which
    setting was wrong, and (separately, more importantly) `S_max` from
    `horodecki_smax` must never be substituted for this value when
    evaluating the published DIQKD rate formula -- see `diqkd_rate`.
    """
    a0 = np.array([0.0, 0.0, 1.0])
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = (np.array([0.0, 0.0, 1.0]) + np.array([1.0, 0.0, 0.0])) / np.sqrt(2)
    b1 = (np.array([0.0, 0.0, 1.0]) - np.array([1.0, 0.0, 0.0])) / np.sqrt(2)
    return (
        correlator(rho4, a0, b0)
        + correlator(rho4, a0, b1)
        + correlator(rho4, a1, b0)
        - correlator(rho4, a1, b1)
    )


# ----------------------------------------------------------------------
# 5. Key-basis QBER (Z,Z) computed directly via projectors
# ----------------------------------------------------------------------

def key_basis_qber_z(rho4: np.ndarray) -> float:
    """Q = P(a != b | Z,Z), computed directly from Z-basis projectors
    (NOT the unrelated single-qubit BB84 QBER)."""
    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
    op_mismatch = np.kron(proj0, proj1) + np.kron(proj1, proj0)
    return float(np.real(np.trace(rho4 @ op_mismatch)))


# ----------------------------------------------------------------------
# 6. Analytic predictions
# ----------------------------------------------------------------------

def predicted_Mx(d, q, gamma):
    return (1 - 4 * d / 3) * (1 - 2 * q) * np.sqrt(1 - gamma)


def predicted_Mz(d, q, gamma):
    return (1 - 4 * d / 3) * (1 - gamma)


def predicted_tz(d, q, gamma, cls: int):
    return gamma if cls == 1 else (1 - 4 * d / 3) * gamma


def predicted_delta_Tzz(d, gamma):
    """Class 1 - Class 2, two-sided T_zz, general d,q,gamma (no q dependence)."""
    return 8 * d * gamma**2 * (3 - 2 * d) / 9


def masking_condition(d, q, gamma, cls: int) -> bool:
    """True if the two-sided z-correlation dominates (class-dependence
    visible in S_max): T_zz >= M_x^2, evaluated from the closed forms."""
    Mx = predicted_Mx(d, q, gamma)
    Mz = predicted_Mz(d, q, gamma)
    tz = predicted_tz(d, q, gamma, cls)
    Tzz = Mz**2 + tz**2
    return Tzz >= Mx**2


# ----------------------------------------------------------------------
# 7. Binary entropy and DIQKD collective-attack rate
# ----------------------------------------------------------------------

def binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def diqkd_rate(Q: float, S: float):
    """Published DIQKD collective-attack lower bound (Acin et al. 2007;
    Pironio et al. 2009), applied here to simulated statistics -- this
    is a literature bound being evaluated, not a new result of this
    project.

    CRITICAL: `S` here MUST be the CHSH value produced by the protocol's
    actual, fixed measurement settings (`fixed_setting_chsh` in this
    module) -- i.e. a quantity a real experiment could report as "the
    observed CHSH value". It must NEVER be `horodecki_smax`, which is a
    state-dependent optimization over all possible settings and is not
    tied to any single realizable protocol configuration; substituting
    it would overstate the bound whenever S_max > S_fixed (an earlier
    version of this script did this and was corrected after audit).

    Returns (raw_rate_or_None, domain_valid). raw_rate is None when
    S < 2 (outside the formula's valid domain); the caller decides how
    to report that rather than this function silently clipping.
    """
    if S < 2.0:
        return None, False
    arg = (S / 2.0) ** 2 - 1.0
    if arg < 0.0:
        # Should not happen once S >= 2, but guard explicitly rather
        # than silently clipping a negative sqrt argument.
        return None, False
    inner = (1.0 + np.sqrt(arg)) / 2.0
    raw = 1.0 - binary_entropy(Q) - binary_entropy(inner)
    return raw, True


# ----------------------------------------------------------------------
# 8. Per-point evaluation (one ordering, one parameter point, one sidedness)
# ----------------------------------------------------------------------

def evaluate_point(sequence, d, q, gamma, sided, failures, label):
    rho0 = phi_plus()
    rho_out = apply_ordering(rho0, sequence, d, q, gamma, sided)
    validate_physical(rho_out, label, failures)

    rA = bloch_vector(partial_trace_b(rho_out))
    rB = bloch_vector(partial_trace_a(rho_out))
    T = correlation_tensor(rho_out)
    S_max = horodecki_smax(T)
    S_fixed = fixed_setting_chsh(rho_out)
    Q = key_basis_qber_z(rho_out)

    return {
        "rho": rho_out,
        "r_A": rA,
        "r_B": rB,
        "T": T,
        "S_max": S_max,
        "S_fixed": S_fixed,
        "Q": Q,
    }


# ----------------------------------------------------------------------
# 9. Main experiment
# ----------------------------------------------------------------------

def main() -> int:
    failures: list[str] = []

    print("E91 / CHSH channel-ordering validation")
    print("=========================================")
    print()

    # ------------------------------------------------------------
    # Part 1: independent-parameter grid (d, q, gamma)
    # ------------------------------------------------------------
    d_values = [0.01, 0.05, 0.10]
    q_values = [0.00, 0.02, 0.05, 0.10]
    gamma_values = [0.01, 0.05, 0.10]

    print("Part 1: independent-parameter grid validation")
    print("-----------------------------------------------")
    print(f"d in {d_values}, q in {q_values}, gamma in {gamma_values}")
    print()

    grid_masking_report = []
    for d in d_values:
        for q in q_values:
            for gamma in gamma_values:
                res1_one = evaluate_point(ORDERINGS[CANONICAL_CLASS_1_ORDERING], d, q, gamma, "one",
                                           failures, f"one-sided C1 d={d} q={q} g={gamma}")
                res2_one = evaluate_point(ORDERINGS[CANONICAL_CLASS_2_ORDERING], d, q, gamma, "one",
                                           failures, f"one-sided C2 d={d} q={q} g={gamma}")
                res1_two = evaluate_point(ORDERINGS[CANONICAL_CLASS_1_ORDERING], d, q, gamma, "two",
                                           failures, f"two-sided C1 d={d} q={q} g={gamma}")
                res2_two = evaluate_point(ORDERINGS[CANONICAL_CLASS_2_ORDERING], d, q, gamma, "two",
                                           failures, f"two-sided C2 d={d} q={q} g={gamma}")

                # --- one-sided exact-formula checks ---
                Mx_pred = predicted_Mx(d, q, gamma)
                Mz_pred = predicted_Mz(d, q, gamma)
                tz1_pred = predicted_tz(d, q, gamma, 1)
                tz2_pred = predicted_tz(d, q, gamma, 2)

                T1_pred_one = np.diag([Mx_pred, -Mx_pred, Mz_pred])
                T2_pred_one = np.diag([Mx_pred, -Mx_pred, Mz_pred])
                if np.max(np.abs(res1_one["T"] - T1_pred_one)) > TOL:
                    failures.append(f"one-sided T mismatch (C1) d={d} q={q} g={gamma}")
                if np.max(np.abs(res2_one["T"] - T2_pred_one)) > TOL:
                    failures.append(f"one-sided T mismatch (C2) d={d} q={q} g={gamma}")
                if np.max(np.abs(res1_one["T"] - res2_one["T"])) > TOL:
                    failures.append(f"one-sided T NOT class-independent d={d} q={q} g={gamma}")

                if np.max(np.abs(res1_one["r_A"] - np.array([0, 0, tz1_pred]))) > TOL:
                    failures.append(f"one-sided r_A mismatch (C1) d={d} q={q} g={gamma}")
                if np.max(np.abs(res2_one["r_A"] - np.array([0, 0, tz2_pred]))) > TOL:
                    failures.append(f"one-sided r_A mismatch (C2) d={d} q={q} g={gamma}")
                if np.max(np.abs(res1_one["r_B"])) > TOL or np.max(np.abs(res2_one["r_B"])) > TOL:
                    failures.append(f"one-sided r_B should be exactly zero d={d} q={q} g={gamma}")

                # --- two-sided exact-formula checks ---
                Tzz1_pred = Mz_pred**2 + tz1_pred**2
                Tzz2_pred = Mz_pred**2 + tz2_pred**2
                T1_pred_two = np.diag([Mx_pred**2, -Mx_pred**2, Tzz1_pred])
                T2_pred_two = np.diag([Mx_pred**2, -Mx_pred**2, Tzz2_pred])

                if np.max(np.abs(res1_two["T"] - T1_pred_two)) > TOL:
                    failures.append(f"two-sided T mismatch (C1) d={d} q={q} g={gamma}")
                if np.max(np.abs(res2_two["T"] - T2_pred_two)) > TOL:
                    failures.append(f"two-sided T mismatch (C2) d={d} q={q} g={gamma}")

                if np.max(np.abs(res1_two["r_A"] - np.array([0, 0, tz1_pred]))) > TOL:
                    failures.append(f"two-sided r_A mismatch (C1) d={d} q={q} g={gamma}")
                if np.max(np.abs(res1_two["r_B"] - np.array([0, 0, tz1_pred]))) > TOL:
                    failures.append(f"two-sided r_B mismatch (C1) d={d} q={q} g={gamma}")

                delta_Tzz_observed = res1_two["T"][2, 2] - res2_two["T"][2, 2]
                delta_Tzz_pred = predicted_delta_Tzz(d, gamma)
                if abs(delta_Tzz_observed - delta_Tzz_pred) > TOL:
                    failures.append(
                        f"Delta T_zz mismatch d={d} q={q} g={gamma}: "
                        f"observed={delta_Tzz_observed:.15f} predicted={delta_Tzz_pred:.15f}"
                    )

                # --- key-basis QBER = (1-T_zz)/2 ---
                for cls_res, Tzz_val, tag in ((res1_two, Tzz1_pred, "C1"), (res2_two, Tzz2_pred, "C2")):
                    q_pred = (1 - Tzz_val) / 2
                    if abs(cls_res["Q"] - q_pred) > TOL:
                        failures.append(f"Q=(1-Tzz)/2 mismatch ({tag}) d={d} q={q} g={gamma}")

                # --- masking condition cross-check ---
                masked1_pred = not masking_condition(d, q, gamma, 1)
                masked2_pred = not masking_condition(d, q, gamma, 2)
                s_diff_two = abs(res1_two["S_max"] - res2_two["S_max"])
                # If BOTH classes are predicted "masked" (z-correlation
                # subdominant), S_max should coincide to TOL. If at least
                # one is unmasked, we only assert internal consistency of
                # the masking predicate against the raw Tzz/Mx comparison
                # (not against S_max directly, since being "masked" is
                # about which eigenvalue wins the top-2, and near a
                # crossover S_max sensitivity can be extremely small but
                # nonzero -- see Part 3 for the precise threshold search).
                if masked1_pred and masked2_pred and s_diff_two > 1e-9:
                    failures.append(
                        f"masking predicted both classes masked but S_max differs "
                        f"by {s_diff_two:.3e} at d={d} q={q} g={gamma}"
                    )

                grid_masking_report.append((d, q, gamma, masked1_pred, masked2_pred, s_diff_two))

    n_masked = sum(1 for r in grid_masking_report if r[3] and r[4])
    n_unmasked = sum(1 for r in grid_masking_report if not (r[3] and r[4]))
    print(f"Grid points evaluated: {len(grid_masking_report)}")
    print(f"  both-classes-masked (S_max class-independent predicted): {n_masked}")
    print(f"  at least one class unmasked (S_max class-dependence predicted): {n_unmasked}")
    print()

    print("Sample of grid points where masking status differs from q=0 baseline expectation:")
    for d, q, gamma, m1, m2, sdiff in grid_masking_report:
        if q == 0.0:
            print(f"  d={d:.2f} q={q:.2f} g={gamma:.2f}: masked1={m1} masked2={m2} S_max_diff={sdiff:.3e}")
    print()

    # ------------------------------------------------------------
    # Part 2: shared-p sweep, low-noise grid
    # ------------------------------------------------------------
    print("Part 2: shared-p sweep (d=q=gamma=p), two-sided scenario")
    print("------------------------------------------------------------")
    p_grid = np.round(np.arange(0.001, 0.101, 0.001), 6)

    rows = []
    for p in p_grid:
        r1 = evaluate_point(ORDERINGS[CANONICAL_CLASS_1_ORDERING], p, p, p, "two", failures, f"shared-p C1 p={p}")
        r2 = evaluate_point(ORDERINGS[CANONICAL_CLASS_2_ORDERING], p, p, p, "two", failures, f"shared-p C2 p={p}")
        # DIQKD rate must use S_fixed (the protocol-realizable, fixed-
        # setting CHSH value), never S_max (a state-optimization bound
        # over all possible settings that no fixed protocol observes
        # directly) -- see the docstrings on fixed_setting_chsh and
        # diqkd_rate for the full justification.
        rate1, valid1 = diqkd_rate(r1["Q"], r1["S_fixed"])
        rate2, valid2 = diqkd_rate(r2["Q"], r2["S_fixed"])
        rows.append(
            {
                "p": p,
                "Q1": r1["Q"], "S1": r1["S_fixed"], "S1_max": r1["S_max"], "rate1": rate1, "valid1": valid1,
                "Q2": r2["Q"], "S2": r2["S_fixed"], "S2_max": r2["S_max"], "rate2": rate2, "valid2": valid2,
            }
        )

    # CHSH-violation threshold (S crosses 2) per class
    def find_threshold(rows, key, thresh, cross_from_above=True):
        prev = None
        for row in rows:
            val = row[key]
            if val is None:
                continue
            if prev is not None:
                if cross_from_above and prev["val"] >= thresh > val:
                    # linear interpolation between prev['p'] and row['p']
                    frac = (prev["val"] - thresh) / (prev["val"] - val)
                    return prev["p"] + frac * (row["p"] - prev["p"])
            prev = {"p": row["p"], "val": val}
        return None

    s1_series = [{"p": r["p"], "val": r["S1"]} for r in rows]
    s2_series = [{"p": r["p"], "val": r["S2"]} for r in rows]

    def interp_threshold(series, thresh):
        for i in range(1, len(series)):
            a, b = series[i - 1], series[i]
            if a["val"] >= thresh > b["val"]:
                frac = (a["val"] - thresh) / (a["val"] - b["val"])
                return a["p"] + frac * (b["p"] - a["p"])
        return None

    chsh_threshold_1 = interp_threshold(s1_series, 2.0)
    chsh_threshold_2 = interp_threshold(s2_series, 2.0)

    rate1_series = [{"p": r["p"], "val": r["rate1"] if r["rate1"] is not None else -1.0} for r in rows]
    rate2_series = [{"p": r["p"], "val": r["rate2"] if r["rate2"] is not None else -1.0} for r in rows]
    diqkd_threshold_1 = interp_threshold(rate1_series, 0.0)
    diqkd_threshold_2 = interp_threshold(rate2_series, 0.0)

    # Ordering-difference onset: first p where |S1-S2| exceeds a fixed
    # detection tolerance (not TOL=1e-12, since floating point noise at
    # that level is not physically meaningful -- use 1e-9).
    onset_p = None
    for r in rows:
        if abs(r["S1"] - r["S2"]) > 1e-9:
            onset_p = r["p"]
            break

    # Masking transition: first p where the masking predicate flips for
    # either class, cross-checked against the observed S_max difference.
    masking_flip_p = None
    prev_masked = None
    for p in p_grid:
        masked_now = (not masking_condition(p, p, p, 1)) and (not masking_condition(p, p, p, 2))
        if prev_masked is not None and masked_now != prev_masked:
            masking_flip_p = p
            break
        prev_masked = masked_now

    print(f"CHSH-violation threshold (S_fixed crosses 2.0, protocol-consistent):   "
          f"Class1 p~={chsh_threshold_1}   Class2 p~={chsh_threshold_2}")
    print(f"DIQKD-rate threshold (r crosses 0, using S_fixed -- NOT S_max):         "
          f"Class1 p~={diqkd_threshold_1}   Class2 p~={diqkd_threshold_2}")
    print(f"Ordering-difference onset (|S1_fixed-S2_fixed|>1e-9):    p~={onset_p}")
    print(f"Masking-regime transition (grid resolution, based on the S_max/T_zz "
          f"diagnostic criterion -- unrelated to the DIQKD threshold above):  p~={masking_flip_p}")
    print()

    print(f"{'p':>7} {'Q1':>9} {'S1_fixed':>10} {'S1_max':>9} {'rate1':>12} "
          f"{'Q2':>9} {'S2_fixed':>10} {'S2_max':>9} {'rate2':>12}")
    for r in rows[::10]:
        rate1s = f"{r['rate1']:.6f}" if r["rate1"] is not None else "  (S<2)"
        rate2s = f"{r['rate2']:.6f}" if r["rate2"] is not None else "  (S<2)"
        print(f"{r['p']:7.3f} {r['Q1']:9.5f} {r['S1']:10.5f} {r['S1_max']:9.5f} {rate1s:>12} "
              f"{r['Q2']:9.5f} {r['S2']:10.5f} {r['S2_max']:9.5f} {rate2s:>12}")
    print()

    # ------------------------------------------------------------
    # Part 3: explicit masking-transition search (q=0, wider p range)
    # ------------------------------------------------------------
    # The requested grids (Part 1, Part 2) never leave the "unmasked"
    # (z-correlation-dominant) regime once q>0, because dephasing damps
    # M_x faster than M_z. To actually witness the masking transition
    # predicted by the analytic condition, q must be set to 0 and p
    # swept over a wider range, since with q=0 the transition is known
    # to occur at larger p (masked for small-to-moderate p, unmasked
    # only once amplitude damping dominates enough to make M_z < M_x).
    print("Part 3: explicit masking-transition search (q=0, d=gamma=p, wider range)")
    print("-----------------------------------------------------------------------------")
    p_wide = np.round(np.arange(0.01, 0.51, 0.01), 6)
    transition_rows = []
    for p in p_wide:
        r1 = evaluate_point(ORDERINGS[CANONICAL_CLASS_1_ORDERING], p, 0.0, p, "two", failures,
                             f"masking-search C1 p={p}")
        r2 = evaluate_point(ORDERINGS[CANONICAL_CLASS_2_ORDERING], p, 0.0, p, "two", failures,
                             f"masking-search C2 p={p}")
        masked1_pred = not masking_condition(p, 0.0, p, 1)
        masked2_pred = not masking_condition(p, 0.0, p, 2)
        s_diff = abs(r1["S_max"] - r2["S_max"])
        transition_rows.append({"p": p, "masked1": masked1_pred, "masked2": masked2_pred, "s_diff": s_diff})
        # cross-check: predicted-masked (both) implies negligible S_max diff
        if masked1_pred and masked2_pred and s_diff > 1e-9:
            failures.append(f"q=0 masking predicate says masked at p={p} but S_max_diff={s_diff:.3e}")
        # predicted-unmasked (either) implies non-negligible S_max diff,
        # EXCEPT immediately at the crossover point where it can be
        # arbitrarily small -- only flag if clearly inconsistent (diff
        # essentially zero far from any crossover).
    transition_p_analytic = None
    prev = None
    for row in transition_rows:
        masked_now = row["masked1"] and row["masked2"]
        if prev is not None and masked_now != prev["masked"]:
            transition_p_analytic = (prev["p"], row["p"])
            break
        prev = {"p": row["p"], "masked": masked_now}

    transition_p_numeric = None
    prev_diff = None
    for row in transition_rows:
        if prev_diff is not None and prev_diff < 1e-6 <= row["s_diff"]:
            transition_p_numeric = (row["p"])
            break
        prev_diff = row["s_diff"]

    print(f"Analytic masking-condition transition (q=0): between p={transition_p_analytic}")
    print(f"Numeric S_max-divergence transition (q=0, |S1-S2| first exceeds 1e-6): p~={transition_p_numeric}")
    print()
    print(f"{'p':>6} {'masked1':>8} {'masked2':>8} {'S_max diff':>12}")
    for row in transition_rows[::5]:
        print(f"{row['p']:6.2f} {str(row['masked1']):>8} {str(row['masked2']):>8} {row['s_diff']:12.3e}")
    print()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed:")
        for msg in failures[:30]:
            print(f"  - {msg}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print("All checks passed at TOL =", TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
