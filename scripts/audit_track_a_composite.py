import math
import numpy as np


# ============================================================
# Track-A literal channel definitions from the paper
# ============================================================

I = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex
)

Y = np.array(
    [[0, -1j],
     [1j, 0]],
    dtype=complex
)

Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex
)


ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

ket_plus = (ket0 + ket1) / math.sqrt(2)
ket_minus = (ket0 - ket1) / math.sqrt(2)


# ============================================================
# Paper Eq. (3): Depolarizing
#
# N_dep(rho) =
#   (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
# ============================================================

def depolarizing(rho, p):
    return (
        (1.0 - p) * rho
        + (p / 3.0) * (
            X @ rho @ X
            + Y @ rho @ Y
            + Z @ rho @ Z
        )
    )


# ============================================================
# Paper Eq. (4): Dephasing / phase-flip
#
# N_deph(rho) =
#   (1-p) rho + p Z rho Z
# ============================================================

def dephasing(rho, p):
    return (
        (1.0 - p) * rho
        + p * Z @ rho @ Z
    )


# ============================================================
# Paper Eq. (5)-(6): Amplitude damping
#
# gamma = p for triple composite
# ============================================================

def amplitude_damping(rho, gamma):
    K0 = np.array(
        [[1.0, 0.0],
         [0.0, math.sqrt(1.0 - gamma)]],
        dtype=complex
    )

    K1 = np.array(
        [[0.0, math.sqrt(gamma)],
         [0.0, 0.0]],
        dtype=complex
    )

    return (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )


# ============================================================
# Composite channels
# ============================================================

def apply_channel(state, p, scenario):
    rho = np.outer(state, state.conj())

    # Single:
    # N_dep
    rho = depolarizing(rho, p)

    if scenario in ("dual", "triple"):
        # Dual:
        # N_deph(N_dep(rho))
        rho = dephasing(rho, p)

    if scenario == "triple":
        # Triple:
        # N_AD(N_deph(N_dep(rho)))
        rho = amplitude_damping(rho, p)

    return rho


# ============================================================
# Measurement error
# ============================================================

def measurement_error(rho, state, bit):
    """
    Determine the BB84 measurement error for the
    corresponding state.

    |0> -> Z basis, bit 0
    |1> -> Z basis, bit 1
    |+> -> X basis, bit 0
    |-> -> X basis, bit 1
    """

    if np.allclose(state, ket0):
        probabilities = [
            np.real(rho[0, 0]),
            np.real(rho[1, 1]),
        ]

    elif np.allclose(state, ket1):
        probabilities = [
            np.real(rho[0, 0]),
            np.real(rho[1, 1]),
        ]

    else:
        p_plus = np.real(
            ket_plus.conj() @ rho @ ket_plus
        )

        p_minus = np.real(
            ket_minus.conj() @ rho @ ket_minus
        )

        probabilities = [
            p_plus,
            p_minus,
        ]

    return probabilities[1 - bit]


# ============================================================
# Exact BB84 QBER
# ============================================================

def exact_bb84_qber(p, scenario):

    states = [
        (ket0, 0),
        (ket1, 1),
        (ket_plus, 0),
        (ket_minus, 1),
    ]

    errors = []

    for state, bit in states:

        rho = apply_channel(
            state,
            p,
            scenario
        )

        error = measurement_error(
            rho,
            state,
            bit
        )

        errors.append(error)

    return float(np.mean(errors))


# ============================================================
# Paper reported values from Table 4
#
# Table 4 gives:
#
# p       QBER Dual   QBER Single
# 0.02      0.064        0.049
# 0.04      0.132        0.099
# 0.06      0.207        0.151
# 0.08      0.287        0.199
# 0.10      0.368        0.246
# ============================================================

paper_table4 = {
    0.02: (0.064, 0.049),
    0.04: (0.132, 0.099),
    0.06: (0.207, 0.151),
    0.08: (0.287, 0.199),
    0.10: (0.368, 0.246),
}


# ============================================================
# Main audit
# ============================================================

print("Track-A composite-channel audit")
print("================================")
print()
print("Literal channel definitions from paper:")
print("  Depolarizing : Eq. (3)")
print("  Dephasing    : Eq. (4), phase-flip")
print("  Amplitude AD : Eq. (5)-(6)")
print("  Dual order   : Dephasing(Depolarizing(rho))")
print("  Triple order : AD(Dephasing(Depolarizing(rho)))")
print()

print(
    f"{'p':>5} "
    f"{'Single':>12} "
    f"{'Dual':>12} "
    f"{'Triple':>12}"
)

print("-" * 48)

for p in np.arange(0.01, 0.101, 0.01):

    p = round(float(p), 2)

    single = exact_bb84_qber(
        p,
        "single"
    )

    dual = exact_bb84_qber(
        p,
        "dual"
    )

    triple = exact_bb84_qber(
        p,
        "triple"
    )

    print(
        f"{p:5.2f} "
        f"{single:12.6f} "
        f"{dual:12.6f} "
        f"{triple:12.6f}"
    )


print()
print("Comparison with paper Table 4")
print("==============================")
print()

for p, (paper_dual, paper_single) in paper_table4.items():

    our_single = exact_bb84_qber(
        p,
        "single"
    )

    our_dual = exact_bb84_qber(
        p,
        "dual"
    )

    print(f"p = {p:.2f}")

    print(
        f"  Single:"
        f" ours={our_single:.6f}"
        f" paper={paper_single:.6f}"
        f" difference={our_single - paper_single:+.6f}"
    )

    print(
        f"  Dual:"
        f" ours={our_dual:.6f}"
        f" paper={paper_dual:.6f}"
        f" difference={our_dual - paper_dual:+.6f}"
    )

    print()


print("Key p = 0.05 diagnostic")
print("========================")

p = 0.05

for scenario in ("single", "dual", "triple"):

    qber = exact_bb84_qber(
        p,
        scenario
    )

    print(
        f"{scenario:>7} QBER = {qber:.6f}"
    )

print()
print("Audit complete.")
