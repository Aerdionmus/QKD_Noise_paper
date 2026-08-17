import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Kraus

from qkd_noise.channels import (
    depolarizing_kraus,
    dephasing_kraus,
    amplitude_damping_kraus,
)


def prepare_bb84_state(bit: int, basis: int) -> QuantumCircuit:
    """
    Prepare one BB84 state.

    basis = 0 -> Z basis
    basis = 1 -> X basis

    bit=0, basis=0 -> |0>
    bit=1, basis=0 -> |1>
    bit=0, basis=1 -> |+>
    bit=1, basis=1 -> |->
    """

    qr = QuantumRegister(1, "q")
    cr = ClassicalRegister(1, "c")

    qc = QuantumCircuit(qr, cr)

    if bit == 1:
        qc.x(0)

    if basis == 1:
        qc.h(0)

    return qc


def add_exact_noise(
    qc: QuantumCircuit,
    p: float,
    scenario: str,
):
    """
    Apply the manuscript's exact Kraus channels.

    single:
        depolarizing

    dual:
        depolarizing -> dephasing

    triple:
        depolarizing -> dephasing -> amplitude damping
    """

    if scenario not in {"single", "dual", "triple"}:
        raise ValueError(
            "scenario must be 'single', 'dual', or 'triple'"
        )

    # Paper Eq. (3)
    dep = Kraus(depolarizing_kraus(p))
    qc.append(dep, [0])

    if scenario in {"dual", "triple"}:
        # Paper Eq. (4)
        deph = Kraus(dephasing_kraus(p))
        qc.append(deph, [0])

    if scenario == "triple":
        # Paper Eq. (6), gamma = p
        ad = Kraus(amplitude_damping_kraus(p))
        qc.append(ad, [0])


def measure_bb84(qc: QuantumCircuit, bob_basis: int):
    """
    Measure in Bob's chosen basis.

    Z basis -> computational measurement
    X basis -> H followed by computational measurement
    """

    if bob_basis == 1:
        qc.h(0)

    qc.measure(0, 0)


def run_single_bb84_round(
    alice_bit: int,
    alice_basis: int,
    bob_basis: int,
    p: float,
    scenario: str,
    shots: int = 1,
    seed: int | None = None,
) -> int:

    qc = prepare_bb84_state(alice_bit, alice_basis)

    add_exact_noise(qc, p, scenario)

    measure_bb84(qc, bob_basis)

    backend = AerSimulator()

    result = backend.run(
        qc,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts()

    # Since shots=1, exactly one outcome is expected.
    outcome = next(iter(counts))

    return int(outcome)


def simulate_bb84(
    n: int = 1000,
    p: float = 0.05,
    scenario: str = "single",
    seed: int | None = 12345,
):
    """
    Simulate n BB84 transmissions.

    Returns:
        dictionary containing Alice/Bob data,
        sifted key and QBER.
    """

    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=n)
    alice_bases = rng.integers(0, 2, size=n)
    bob_bases = rng.integers(0, 2, size=n)

    bob_bits = np.empty(n, dtype=int)

    for i in range(n):

        bob_bits[i] = run_single_bb84_round(
            alice_bit=int(alice_bits[i]),
            alice_basis=int(alice_bases[i]),
            bob_basis=int(bob_bases[i]),
            p=p,
            scenario=scenario,
            shots=1,
            seed=int(rng.integers(0, 2**32 - 1)),
        )

    # Sifting: retain only matching bases.
    sift_mask = alice_bases == bob_bases

    alice_sifted = alice_bits[sift_mask]
    bob_sifted = bob_bits[sift_mask]

    if len(alice_sifted) == 0:
        qber = 0.0
    else:
        qber = float(
            np.mean(alice_sifted != bob_sifted)
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
        "sifted_length": len(alice_sifted),
    }
