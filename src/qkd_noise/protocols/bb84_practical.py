import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Kraus

from qkd_noise.channels import (
    depolarizing_kraus,
    dephasing_kraus,
    amplitude_damping_kraus,
)

from qkd_noise.practical import apply_alignment_error


def build_bb84_practical_circuit(
    alice_bit: int,
    alice_basis: int,
    bob_basis: int,
    p: float,
    scenario: str,
    misalignment_deg: float = 2.0,
) -> QuantumCircuit:
    """
    Build one BB84 circuit with the paper's stated quantum
    channels plus our explicit physical misalignment model.

    Quantum channel:
        single = depolarizing
        dual   = depolarizing -> dephasing
        triple = depolarizing -> dephasing
                 -> amplitude damping

    Practical model:
        Ry(2*epsilon) immediately before Bob's basis
        measurement, where epsilon is the specified
        misalignment angle.

    IMPORTANT:
        The paper provides Q_align = sin^2(epsilon),
        but does not explicitly specify this exact Ry
        circuit realization. This is therefore an
        implementation choice for our reproduction study.
    """

    if scenario not in {"single", "dual", "triple"}:
        raise ValueError(
            "scenario must be 'single', 'dual', or 'triple'"
        )

    qc = QuantumCircuit(1, 1)

    # ---------------------------------------------------------
    # Alice: BB84 state preparation
    # ---------------------------------------------------------

    if alice_bit == 1:
        qc.x(0)

    if alice_basis == 1:
        qc.h(0)

    # ---------------------------------------------------------
    # Exact quantum channels from the paper
    # ---------------------------------------------------------

    qc.append(
        Kraus(depolarizing_kraus(p)),
        [0],
    )

    if scenario in {"dual", "triple"}:
        qc.append(
            Kraus(dephasing_kraus(p)),
            [0],
        )

    if scenario == "triple":
        qc.append(
            Kraus(amplitude_damping_kraus(p)),
            [0],
        )

    # ---------------------------------------------------------
    # Our explicit misalignment model
    # ---------------------------------------------------------

    apply_alignment_error(
        qc,
        misalignment_deg,
    )

    # ---------------------------------------------------------
    # Bob: measurement basis
    # ---------------------------------------------------------

    if bob_basis == 1:
        qc.h(0)

    qc.measure(0, 0)

    return qc


def simulate_bb84_practical(
    n: int = 1000,
    p: float = 0.05,
    scenario: str = "single",
    misalignment_deg: float = 2.0,
    seed: int | None = 12345,
):
    """
    Run n BB84 transmissions with quantum noise plus
    the explicit alignment model.
    """

    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=n)
    alice_bases = rng.integers(0, 2, size=n)
    bob_bases = rng.integers(0, 2, size=n)

    circuits = []

    for i in range(n):

        qc = build_bb84_practical_circuit(
            alice_bit=int(alice_bits[i]),
            alice_basis=int(alice_bases[i]),
            bob_basis=int(bob_bases[i]),
            p=p,
            scenario=scenario,
            misalignment_deg=misalignment_deg,
        )

        circuits.append(qc)

    backend = AerSimulator()

    result = backend.run(
        circuits,
        shots=1,
        seed_simulator=seed,
    ).result()

    bob_bits = np.empty(
        n,
        dtype=int,
    )

    for i in range(n):

        counts = result.get_counts(i)

        outcome = next(iter(counts))

        bob_bits[i] = int(outcome)

    # ---------------------------------------------------------
    # Sifting
    # ---------------------------------------------------------

    sift_mask = (
        alice_bases == bob_bases
    )

    alice_sifted = (
        alice_bits[sift_mask]
    )

    bob_sifted = (
        bob_bits[sift_mask]
    )

    if len(alice_sifted) == 0:
        qber = 0.0
    else:
        qber = float(
            np.mean(
                alice_sifted != bob_sifted
            )
        )

    return {
        "alice_bits": alice_bits,
        "alice_bases": alice_bases,
        "bob_bases": bob_bases,
        "bob_bits": bob_bits,
        "sift_mask": sift_mask,
        "alice_sifted": alice_sifted,
        "bob_sifted": bob_sifted,
        "qber": qber,
        "sifted_length": len(
            alice_sifted
        ),
        "total_bits": n,
    }
