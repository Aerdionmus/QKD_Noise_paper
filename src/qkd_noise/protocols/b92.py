"""
B92 quantum key-distribution protocol.

B92 uses two non-orthogonal states:

    bit 0 -> |0>
    bit 1 -> |+>

Bob randomly chooses between two measurements:

    Z measurement:
        outcome |1> is conclusive -> infer bit 1

    X measurement:
        outcome |-> is conclusive -> infer bit 0

All other outcomes are inconclusive and are discarded.

Noise scenarios:

    single = depolarizing
    dual   = depolarizing -> dephasing
    triple = depolarizing -> dephasing -> amplitude damping
"""

import numpy as np

from qkd_noise.channels import (
    depolarizing_kraus,
    dephasing_kraus,
    amplitude_damping_kraus,
)


# ---------------------------------------------------------------------
# B92 states
# ---------------------------------------------------------------------

SQRT2_INV = 1.0 / np.sqrt(2.0)

KET_0 = np.array(
    [1.0, 0.0],
    dtype=complex,
)

KET_PLUS = np.array(
    [SQRT2_INV, SQRT2_INV],
    dtype=complex,
)


# ---------------------------------------------------------------------
# Density-matrix utilities
# ---------------------------------------------------------------------

def density_matrix(state):
    """Return |psi><psi|."""
    state = np.asarray(state, dtype=complex)

    return np.outer(
        state,
        np.conjugate(state),
    )


def apply_kraus(rho, kraus_operators):
    """Apply a quantum channel represented by Kraus operators."""

    result = np.zeros_like(
        rho,
        dtype=complex,
    )

    for operator in kraus_operators:
        result += (
            operator
            @ rho
            @ operator.conj().T
        )

    return result


# ---------------------------------------------------------------------
# Channel application
# ---------------------------------------------------------------------

def apply_noise(rho, p, scenario):
    """
    Apply the Track-A channel sequence.

    single:
        depolarizing

    dual:
        depolarizing -> dephasing

    triple:
        depolarizing -> dephasing -> amplitude damping
    """

    if scenario not in {
        "single",
        "dual",
        "triple",
    }:
        raise ValueError(
            "scenario must be 'single', 'dual', or 'triple'"
        )

    rho = apply_kraus(
        rho,
        depolarizing_kraus(p),
    )

    if scenario in {
        "dual",
        "triple",
    }:
        rho = apply_kraus(
            rho,
            dephasing_kraus(p),
        )

    if scenario == "triple":
        rho = apply_kraus(
            rho,
            amplitude_damping_kraus(p),
        )

    return rho


# ---------------------------------------------------------------------
# Measurement probabilities
# ---------------------------------------------------------------------

def measurement_probability(rho, projector):
    """Return Tr(P rho) for a measurement projector."""

    probability = np.trace(
        projector @ rho
    )

    return float(
        np.real_if_close(probability)
    )


def z_measurement_probabilities(rho):
    """
    Z-basis measurement.

    Returns:
        P(0), P(1)
    """

    p0 = measurement_probability(
        rho,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=complex,
        ),
    )

    p1 = measurement_probability(
        rho,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=complex,
        ),
    )

    return p0, p1


def x_measurement_probabilities(rho):
    """
    X-basis measurement.

    Returns:
        P(+), P(-)
    """

    ket_plus = KET_PLUS

    ket_minus = np.array(
        [SQRT2_INV, -SQRT2_INV],
        dtype=complex,
    )

    plus_projector = density_matrix(
        ket_plus
    )

    minus_projector = density_matrix(
        ket_minus
    )

    p_plus = measurement_probability(
        rho,
        plus_projector,
    )

    p_minus = measurement_probability(
        rho,
        minus_projector,
    )

    return p_plus, p_minus


# ---------------------------------------------------------------------
# Single B92 transmission
# ---------------------------------------------------------------------

def simulate_transmission(
    alice_bit,
    bob_basis,
    p=0.0,
    scenario="single",
    rng=None,
):
    """
    Simulate one B92 transmission.

    Alice:
        bit 0 -> |0>
        bit 1 -> |+>

    Bob:
        basis 0 -> Z measurement
        basis 1 -> X measurement

    Conclusive events:

        Bob Z + outcome 1:
            infer Alice bit 1

        Bob X + outcome -:
            infer Alice bit 0

    Returns a dictionary containing the raw measurement
    and whether the result is conclusive.
    """

    if alice_bit not in {0, 1}:
        raise ValueError(
            "alice_bit must be 0 or 1"
        )

    if bob_basis not in {0, 1}:
        raise ValueError(
            "bob_basis must be 0 or 1"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must be between 0 and 1"
        )

    if rng is None:
        rng = np.random.default_rng()

    if alice_bit == 0:
        state = KET_0
    else:
        state = KET_PLUS

    rho = density_matrix(state)

    rho = apply_noise(
        rho,
        p,
        scenario,
    )

    # -------------------------------------------------------------
    # Bob's measurement
    # -------------------------------------------------------------

    if bob_basis == 0:
        # Z basis
        p0, p1 = z_measurement_probabilities(rho)

        outcome = int(
            rng.random() >= p0
        )

        # Z outcome |1> is conclusive.
        if outcome == 1:
            conclusive = True
            bob_bit = 1
        else:
            conclusive = False
            bob_bit = None

    else:
        # X basis
        p_plus, p_minus = x_measurement_probabilities(rho)

        outcome = int(
            rng.random() >= p_plus
        )

        # X outcome |-> is conclusive.
        if outcome == 1:
            conclusive = True
            bob_bit = 0
        else:
            conclusive = False
            bob_bit = None

    return {
        "alice_bit": alice_bit,
        "bob_basis": bob_basis,
        "outcome": outcome,
        "conclusive": conclusive,
        "bob_bit": bob_bit,
    }


# ---------------------------------------------------------------------
# Batch B92 simulation
# ---------------------------------------------------------------------

def simulate_b92(
    n=1000,
    p=0.0,
    scenario="single",
    seed=12345,
):
    """
    Run n independent B92 transmissions.

    Returns:
        Alice data,
        Bob data,
        conclusive mask,
        sifted key,
        QBER,
        conclusive count,
        conclusive fraction.
    """

    if n <= 0:
        raise ValueError(
            "n must be positive"
        )

    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(
        0,
        2,
        size=n,
    )

    bob_bases = rng.integers(
        0,
        2,
        size=n,
    )

    bob_bits = np.full(
        n,
        -1,
        dtype=int,
    )

    conclusive = np.zeros(
        n,
        dtype=bool,
    )

    outcomes = np.full(
        n,
        -1,
        dtype=int,
    )

    for i in range(n):

        result = simulate_transmission(
            alice_bit=int(alice_bits[i]),
            bob_basis=int(bob_bases[i]),
            p=p,
            scenario=scenario,
            rng=rng,
        )

        outcomes[i] = result["outcome"]

        if result["conclusive"]:
            conclusive[i] = True
            bob_bits[i] = result["bob_bit"]

    # -------------------------------------------------------------
    # Sifted B92 key
    # -------------------------------------------------------------

    alice_sifted = alice_bits[
        conclusive
    ]

    bob_sifted = bob_bits[
        conclusive
    ]

    conclusive_count = int(
        np.sum(conclusive)
    )

    conclusive_fraction = (
        conclusive_count / n
    )

    # -------------------------------------------------------------
    # QBER
    # -------------------------------------------------------------

    if conclusive_count == 0:
        qber = 0.0
    else:
        qber = float(
            np.mean(
                alice_sifted != bob_sifted
            )
        )

    return {
        "alice_bits": alice_bits,
        "bob_bases": bob_bases,
        "outcomes": outcomes,
        "bob_bits": bob_bits,
        "conclusive_mask": conclusive,
        "alice_sifted": alice_sifted,
        "bob_sifted": bob_sifted,
        "qber": qber,
        "conclusive_count": conclusive_count,
        "conclusive_fraction": conclusive_fraction,
        "total_bits": n,
    }
