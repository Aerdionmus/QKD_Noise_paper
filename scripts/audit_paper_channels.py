"""
Track A — Paper channel-definition audit.

Purpose:
    Record the three composite-noise configurations exactly as
    defined in the paper before implementing the full QKD
    simulation.

Paper:
    Composite Noise Interaction in Discrete-Variable QKD

Scenarios:
    1. Single depolarizing
    2. Dual depolarizing -> dephasing
    3. Triple depolarizing -> dephasing -> amplitude damping

Important:
    This script does NOT attempt to "correct" the paper.
    It records the paper configuration and checks the resulting
    one-qubit channels using density matrices.
"""

import numpy as np


# ============================================================
# Configuration
# ============================================================

P_VALUES = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
]


# ============================================================
# Basic operators
# ============================================================

I = np.eye(2, dtype=complex)

X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Y = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)


ZERO = np.array(
    [
        [1],
        [0],
    ],
    dtype=complex,
)

ONE = np.array(
    [
        [0],
        [1],
    ],
    dtype=complex,
)


def ket0():
    return ZERO.copy()


def ket1():
    return ONE.copy()


def density(ket):
    return ket @ ket.conj().T


# ============================================================
# Paper noise channels
# ============================================================

def depolarizing_channel(rho, p):
    """
    Standard single-qubit depolarizing channel:

        N_dep(rho)
          = (1-p) rho
            + p/3 (X rho X + Y rho Y + Z rho Z)

    The paper labels this as the depolarizing channel used
    in Scenario 1.

    NOTE:
        We isolate this definition here so it can later be
        compared directly against the implementation used by
        Qiskit.
    """

    return (
        (1.0 - p) * rho
        + (p / 3.0)
        * (
            X @ rho @ X
            + Y @ rho @ Y
            + Z @ rho @ Z
        )
    )


def dephasing_channel(rho, p):
    """
    Standard phase-flip/dephasing representation:

        N_deph(rho)
          = (1-p) rho + p Z rho Z

    This is the convention used by this audit implementation.

    IMPORTANT:
        The paper's displayed noise-configuration section gives
        the channel composition but does not fully spell out the
        Kraus operators in the retrieved text. Therefore this
        convention is explicitly marked as an implementation
        assumption rather than claimed as a verbatim paper
        equation.
    """

    return (
        (1.0 - p) * rho
        + p * (Z @ rho @ Z)
    )


def amplitude_damping_channel(rho, gamma):
    """
    Standard amplitude-damping channel:

        E0 = [[1, 0],
              [0, sqrt(1-gamma)]]

        E1 = [[0, sqrt(gamma)],
              [0, 0]]

        N_AD(rho)
          = E0 rho E0† + E1 rho E1†
    """

    e0 = np.array(
        [
            [1, 0],
            [0, np.sqrt(1.0 - gamma)],
        ],
        dtype=complex,
    )

    e1 = np.array(
        [
            [0, np.sqrt(gamma)],
            [0, 0],
        ],
        dtype=complex,
    )

    return (
        e0 @ rho @ e0.conj().T
        + e1 @ rho @ e1.conj().T
    )


# ============================================================
# Composite channels
# ============================================================

def single_noise(rho, p):
    """
    Paper Scenario 1:

        N_dep(rho)
    """
    return depolarizing_channel(rho, p)


def dual_noise(rho, p):
    """
    Paper Scenario 2:

        N_deph(N_dep(rho))
    """
    state = depolarizing_channel(rho, p)
    state = dephasing_channel(state, p)

    return state


def triple_noise(rho, p):
    """
    Paper Scenario 3:

        N_AD(N_deph(N_dep(rho)))

    with gamma = p.
    """
    state = depolarizing_channel(rho, p)
    state = dephasing_channel(state, p)
    state = amplitude_damping_channel(
        state,
        gamma=p,
    )

    return state


# ============================================================
# Validation helpers
# ============================================================

def trace_of(rho):
    return np.trace(rho)


def hermitian_error(rho):
    return np.max(
        np.abs(
            rho - rho.conj().T
        )
    )


def eigenvalues(rho):
    return np.linalg.eigvalsh(rho)


def validate_density_matrix(rho):
    tr = trace_of(rho)
    herm_err = hermitian_error(rho)
    eigs = eigenvalues(rho)

    trace_ok = np.isclose(
        tr,
        1.0,
        atol=1e-10,
    )

    hermitian_ok = (
        herm_err < 1e-10
    )

    positive_ok = np.min(eigs) >= -1e-10

    return (
        trace_ok,
        hermitian_ok,
        positive_ok,
        tr,
        herm_err,
        eigs,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("Paper Channel Audit")
    print("===================")
    print()

    print("Paper-defined scenarios:")
    print()
    print("Scenario 1:")
    print("    N_dep(rho)")
    print()
    print("Scenario 2:")
    print("    N_deph(N_dep(rho))")
    print()
    print("Scenario 3:")
    print("    N_AD(N_deph(N_dep(rho)))")
    print()
    print("Triple channel uses:")
    print("    gamma = p")
    print()

    # --------------------------------------------------------
    # Test states
    # --------------------------------------------------------

    states = {
        "|0>": density(ket0()),
        "|1>": density(ket1()),
    }

    for p in P_VALUES:

        print(
            f"p = {p:.2f}"
        )
        print(
            "-" * 60
        )

        for state_name, rho in states.items():

            outputs = {
                "single": single_noise(
                    rho,
                    p,
                ),
                "dual": dual_noise(
                    rho,
                    p,
                ),
                "triple": triple_noise(
                    rho,
                    p,
                ),
            }

            print(
                f"Input state: {state_name}"
            )

            for scenario, output in outputs.items():

                (
                    trace_ok,
                    hermitian_ok,
                    positive_ok,
                    tr,
                    herm_err,
                    eigs,
                ) = validate_density_matrix(
                    output
                )

                print(
                    f"  {scenario:<8}"
                    f" trace={tr.real:.10f}"
                    f" herm_err={herm_err:.2e}"
                    f" min_eig={np.min(eigs):.6e}"
                    f" valid="
                    f"{trace_ok and hermitian_ok and positive_ok}"
                )

        print()

    print("Channel audit complete.")
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The dephasing convention used here is explicitly"
    )
    print(
        "marked as an implementation assumption because"
    )
    print(
        "the retrieved paper section specifies the composition"
    )
    print(
        "but does not fully reproduce its Kraus operators."
    )


if __name__ == "__main__":
    main()
