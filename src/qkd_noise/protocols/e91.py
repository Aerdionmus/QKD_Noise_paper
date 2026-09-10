"""
E91 QKD protocol implementation.

Based on:
"Composite Noise Interaction in Discrete-Variable QKD"

Paper parameters:
    - Bell state: |Phi+> = (|00> + |11>) / sqrt(2)
    - Alice bases: 0°, 45°, 90°
    - Bob bases: 45°, 90°, 135°
    - Werner-state visibility: v = 0.97
    - Noise:
        * depolarizing
        * dephasing
        * amplitude damping
    - Noise parameter p in [0.01, 0.10]

The paper does not explicitly specify the four measurement settings
used for the CHSH calculation. Therefore the CHSH setting mapping is
kept explicit and configurable below.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paper parameters
# ---------------------------------------------------------------------------

E91_VISIBILITY = 0.97

ALICE_BASES = (0.0, 45.0, 90.0)
BOB_BASES = (45.0, 90.0, 135.0)

E91_QBER_THRESHOLD = 0.146


# ---------------------------------------------------------------------------
# CHSH configuration
# ---------------------------------------------------------------------------
#
# The executable implementation uses the CHSH convention
#
#     S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|
#
# with the following measurement angles:
#
#     a  = 0°
#     a' = 45°
#     b  = 22.5°
#     b' = -22.5°
#
# The measurement convention is
#
#     M(theta) = cos(2 theta) Z + sin(2 theta) X
# ---------------------------------------------------------------------------
CHSH_SETTINGS = {
    "a": 0.0,
    "a_prime": 45.0,
    "b": 22.5,
    "b_prime": -22.5,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class E91Result:
    """Result of an E91 simulation."""

    qber: float
    chsh_s: float
    sifted_key: List[int]
    sifted_length: int
    secure: bool


# ---------------------------------------------------------------------------
# Basic quantum states
# ---------------------------------------------------------------------------

def phi_plus() -> np.ndarray:
    """
    Return the |Phi+> Bell-state density matrix.

        |Phi+> = (|00> + |11>) / sqrt(2)
    """
    state = np.array(
        [
            1.0 / sqrt(2.0),
            0.0,
            0.0,
            1.0 / sqrt(2.0),
        ],
        dtype=complex,
    )

    return np.outer(state, state.conjugate())


def identity_2() -> np.ndarray:
    """2x2 identity matrix."""
    return np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Werner state
# ---------------------------------------------------------------------------

def werner_state(visibility: float = E91_VISIBILITY) -> np.ndarray:
    """
    Construct the Werner state used in the paper:

        rho_W = v |Phi+><Phi+| + (1-v) I/4

    The paper uses v = 0.97.
    """
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must be between 0 and 1")

    return (
        visibility * phi_plus()
        + (1.0 - visibility) * np.eye(4, dtype=complex) / 4.0
    )


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------

def apply_kraus_channel(
    rho: np.ndarray,
    kraus_operators: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Apply a single-qubit channel to Alice's qubit.

    For a two-qubit state:

        rho' = sum_i (K_i ⊗ I) rho (K_i† ⊗ I)
    """
    result = np.zeros_like(rho, dtype=complex)

    for k in kraus_operators:
        operator = np.kron(k, I)
        result += operator @ rho @ operator.conj().T

    return result


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Depolarizing channel used in the paper:

        N_dep(rho)
          = (1-p)rho
            + p/3 (X rho X + Y rho Y + Z rho Z)

    """
    _validate_probability(p)

    return [
        sqrt(1.0 - p) * I,
        sqrt(p / 3.0) * X,
        sqrt(p / 3.0) * Y,
        sqrt(p / 3.0) * Z,
    ]


def dephasing_kraus(p: float) -> List[np.ndarray]:
    """
    Dephasing channel used in the paper:

        N_deph(rho) = (1-p)rho + p Z rho Z
    """
    _validate_probability(p)

    return [
        sqrt(1.0 - p) * I,
        sqrt(p) * Z,
    ]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Amplitude-damping channel.

        K0 = [[1, 0],
              [0, sqrt(1-gamma)]]

        K1 = [[0, sqrt(gamma)],
              [0, 0]]

    For the paper's triple-noise experiment, gamma is taken
    to be equal to p.
    """
    _validate_probability(gamma)

    k0 = np.array(
        [
            [1.0, 0.0],
            [0.0, sqrt(1.0 - gamma)],
        ],
        dtype=complex,
    )

    k1 = np.array(
        [
            [0.0, sqrt(gamma)],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    return [k0, k1]


def apply_single_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing noise."""
    return apply_kraus_channel(
        rho,
        depolarizing_kraus(p),
    )


def apply_dual_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Apply dual composite noise:

        depolarizing -> dephasing
    """
    rho = apply_kraus_channel(
        rho,
        depolarizing_kraus(p),
    )

    rho = apply_kraus_channel(
        rho,
        dephasing_kraus(p),
    )

    return rho


def apply_triple_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Apply triple composite noise:

        depolarizing -> dephasing -> amplitude damping

    The paper treats these channels sequentially.
    """
    rho = apply_kraus_channel(
        rho,
        depolarizing_kraus(p),
    )

    rho = apply_kraus_channel(
        rho,
        dephasing_kraus(p),
    )

    rho = apply_kraus_channel(
        rho,
        amplitude_damping_kraus(p),
    )

    return rho


def apply_noise(
    rho: np.ndarray,
    p: float,
    scenario: str = "single",
) -> np.ndarray:
    """
    Apply the requested noise scenario.

    Parameters
    ----------
    rho:
        Two-qubit density matrix.

    p:
        Noise probability.

    scenario:
        "none", "single", "dual", or "triple".
    """
    scenario = scenario.lower()

    if scenario == "none":
        return rho.copy()

    if scenario == "single":
        return apply_single_noise(rho, p)

    if scenario == "dual":
        return apply_dual_noise(rho, p)

    if scenario == "triple":
        return apply_triple_noise(rho, p)

    raise ValueError(
        "scenario must be one of: none, single, dual, triple"
    )


# ---------------------------------------------------------------------------
# Measurement operators
# ---------------------------------------------------------------------------

def measurement_observable(angle_degrees: float) -> np.ndarray:
    """
    Construct the ±1 observable for a linear-polarization measurement
    at the requested angle.

    Observable:

        M(theta) = cos(2 theta) Z + sin(2 theta) X

    This provides the basis-angle representation used by the E91
    measurement model.
    """
    theta = np.deg2rad(angle_degrees)

    return (
        cos(2.0 * theta) * Z
        + sin(2.0 * theta) * X
    )


def projectors(angle_degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the +1 and -1 projectors for a measurement angle.
    """
    observable = measurement_observable(angle_degrees)

    plus = (I + observable) / 2.0
    minus = (I - observable) / 2.0

    return plus, minus


# ---------------------------------------------------------------------------
# Correlation calculation
# ---------------------------------------------------------------------------

def correlation(
    rho: np.ndarray,
    alice_angle: float,
    bob_angle: float,
) -> float:
    """
    Calculate E(a,b):

        E(a,b) = Tr[rho (M_a ⊗ M_b)]

    The result lies in [-1, 1].
    """
    alice_operator = measurement_observable(alice_angle)
    bob_operator = measurement_observable(bob_angle)

    operator = np.kron(
        alice_operator,
        bob_operator,
    )

    value = np.trace(rho @ operator)

    return float(np.real_if_close(value))


# ---------------------------------------------------------------------------
# CHSH
# ---------------------------------------------------------------------------
def chsh_value(
    rho: np.ndarray,
    settings: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate the CHSH Bell parameter.

    For the Phi+ Bell state and the observable convention used here,
    use the equivalent CHSH form:

        S = |E(a,b)
             + E(a,b')
             + E(a',b)
             - E(a',b')|
    """
    if settings is None:
        settings = CHSH_SETTINGS

    a = settings["a"]
    a_prime = settings["a_prime"]
    b = settings["b"]
    b_prime = settings["b_prime"]

    e_ab = correlation(rho, a, b)
    e_ab_prime = correlation(rho, a, b_prime)
    e_a_prime_b = correlation(rho, a_prime, b)
    e_a_prime_b_prime = correlation(
        rho,
        a_prime,
        b_prime,
    )

    return abs(
        e_ab
        + e_ab_prime
        + e_a_prime_b
        - e_a_prime_b_prime
    )


# ---------------------------------------------------------------------------
# Measurement sampling
# ---------------------------------------------------------------------------

def sample_measurement(
    rho: np.ndarray,
    alice_angle: float,
    bob_angle: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Sample one pair of E91 measurement outcomes.

    Returns
    -------
    (alice_bit, bob_bit)

    Bits are represented as:
        +1 measurement -> 0
        -1 measurement -> 1
    """
    alice_plus, alice_minus = projectors(alice_angle)
    bob_plus, bob_minus = projectors(bob_angle)

    projectors_a = {
        0: alice_plus,
        1: alice_minus,
    }

    projectors_b = {
        0: bob_plus,
        1: bob_minus,
    }

    probabilities = np.zeros((2, 2), dtype=float)

    for alice_bit in (0, 1):
        for bob_bit in (0, 1):
            projector = np.kron(
                projectors_a[alice_bit],
                projectors_b[bob_bit],
            )

            probability = np.trace(
                projector @ rho
            )

            probabilities[alice_bit, bob_bit] = max(
                0.0,
                float(np.real_if_close(probability)),
            )

    total = probabilities.sum()

    if total <= 0.0:
        raise RuntimeError(
            "Invalid measurement probability distribution."
        )

    probabilities /= total

    flat = probabilities.reshape(-1)

    outcome = rng.choice(
        4,
        p=flat,
    )

    return (
        outcome // 2,
        outcome % 2,
    )


# ---------------------------------------------------------------------------
# Basis selection
# ---------------------------------------------------------------------------

def random_basis_pair(
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Randomly select Alice and Bob measurement bases from the
    sets specified by the paper.
    """
    alice_angle = float(
        rng.choice(ALICE_BASES)
    )

    bob_angle = float(
        rng.choice(BOB_BASES)
    )

    return alice_angle, bob_angle


def is_key_basis_pair(
    alice_angle: float,
    bob_angle: float,
) -> bool:
    """
    Identify a key-generating basis pair.

    The paper states that correlated outcomes where the bases
    coincide form the raw key.

    Because Alice's and Bob's published basis sets are offset
    ({0,45,90} vs {45,90,135}), the only exact shared basis
    angles are 45° and 90°.
    """
    return np.isclose(
        alice_angle,
        bob_angle,
    )


# ---------------------------------------------------------------------------
# QBER
# ---------------------------------------------------------------------------

def calculate_qber(
    alice_bits: Sequence[int],
    bob_bits: Sequence[int],
) -> float:
    """
    Calculate QBER from two equal-length key sequences.
    """
    if len(alice_bits) != len(bob_bits):
        raise ValueError(
            "Alice and Bob key lengths must match."
        )

    if len(alice_bits) == 0:
        return 0.0

    errors = sum(
        a != b
        for a, b in zip(alice_bits, bob_bits)
    )

    return errors / len(alice_bits)


# ---------------------------------------------------------------------------
# Full E91 simulation
# ---------------------------------------------------------------------------

def simulate_e91(
    num_pairs: int = 1000,
    p: float = 0.0,
    scenario: str = "none",
    visibility: float = E91_VISIBILITY,
    seed: Optional[int] = None,
) -> E91Result:
    """
    Run an E91 simulation.

    Parameters
    ----------
    num_pairs:
        Number of entangled pairs generated.

    p:
        Noise probability.

    scenario:
        "none", "single", "dual", or "triple".

    visibility:
        Werner-state visibility. Paper value = 0.97.

    seed:
        Optional random seed.

    Returns
    -------
    E91Result
    """
    if num_pairs <= 0:
        raise ValueError(
            "num_pairs must be positive."
        )

    _validate_probability(p)

    rng = np.random.default_rng(seed)

    # Prepare Werner state.
    rho = werner_state(
        visibility=visibility,
    )

    # Apply the requested channel composition once to the
    # transmitted qubit state before measurement sampling.
    noisy_rho = apply_noise(
        rho,
        p,
        scenario,
    )

    alice_key: List[int] = []
    bob_key: List[int] = []

    for _ in range(num_pairs):
        alice_angle, bob_angle = random_basis_pair(rng)

        alice_bit, bob_bit = sample_measurement(
            noisy_rho,
            alice_angle,
            bob_angle,
            rng,
        )

        if is_key_basis_pair(
            alice_angle,
            bob_angle,
        ):
            alice_key.append(alice_bit)
            bob_key.append(bob_bit)

    qber = calculate_qber(
        alice_key,
        bob_key,
    )

    s_value = chsh_value(noisy_rho)

    return E91Result(
        qber=qber,
        chsh_s=s_value,
        sifted_key=alice_key,
        sifted_length=len(alice_key),
        secure=qber < E91_QBER_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def run_e91_noise_sweep(
    probabilities: Iterable[float],
    scenario: str,
    num_pairs: int = 1000,
    visibility: float = E91_VISIBILITY,
    seed: Optional[int] = None,
) -> List[E91Result]:
    """
    Run E91 for multiple noise probabilities.
    """
    results: List[E91Result] = []

    for index, p in enumerate(probabilities):
        current_seed = (
            None
            if seed is None
            else seed + index
        )

        result = simulate_e91(
            num_pairs=num_pairs,
            p=float(p),
            scenario=scenario,
            visibility=visibility,
            seed=current_seed,
        )

        results.append(result)

    return results


def _validate_probability(p: float) -> None:
    """Validate a channel probability."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "Noise probability must be between 0 and 1."
        )


# ---------------------------------------------------------------------------
# Basic self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("E91 protocol self-test")
    print("=" * 40)

    # Ideal Bell state.
    ideal = phi_plus()

    print(
        f"Ideal |Phi+> CHSH S: "
        f"{chsh_value(ideal):.6f}"
    )

    # Paper's imperfect source.
    werner = werner_state(
        E91_VISIBILITY
    )

    print(
        f"Werner v={E91_VISIBILITY} CHSH S: "
        f"{chsh_value(werner):.6f}"
    )

    print()

    for scenario in (
        "single",
        "dual",
        "triple",
    ):
        result = simulate_e91(
            num_pairs=1000,
            p=0.05,
            scenario=scenario,
            visibility=E91_VISIBILITY,
            seed=42,
        )

        print(
            f"{scenario:>7} | "
            f"QBER={result.qber:.4f} | "
            f"S={result.chsh_s:.4f} | "
            f"sifted={result.sifted_length:4d} | "
            f"secure={result.secure}"
        )