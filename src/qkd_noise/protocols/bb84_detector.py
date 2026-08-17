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

from qkd_noise.detector import detect_bit


def build_bb84_quantum_circuit(
    alice_bit: int,
    alice_basis: int,
    bob_basis: int,
    p: float,
    scenario: str,
    misalignment_deg: float = 2.0,
) -> QuantumCircuit:
    """
    Build the quantum part of one BB84 transmission.

    Noise:
        single = depolarizing
        dual   = depolarizing -> dephasing
        triple = depolarizing -> dephasing
                 -> amplitude damping

    Alignment:
        explicit Ry(2*epsilon) model.
    """

    if scenario not in {
        "single",
        "dual",
        "triple",
    }:
        raise ValueError(
            "scenario must be 'single', 'dual', or 'triple'"
        )

    qc = QuantumCircuit(1, 1)

    # ---------------------------------------------------------
    # Alice: prepare BB84 state
    # ---------------------------------------------------------

    if alice_bit == 1:
        qc.x(0)

    if alice_basis == 1:
        qc.h(0)

    # ---------------------------------------------------------
    # Quantum noise
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
    # Explicit alignment model
    # ---------------------------------------------------------

    apply_alignment_error(
        qc,
        misalignment_deg,
    )

    # ---------------------------------------------------------
    # Bob basis rotation
    # ---------------------------------------------------------

    if bob_basis == 1:
        qc.h(0)

    qc.measure(0, 0)

    return qc


def simulate_bb84_detector(
    n: int = 1000,
    p: float = 0.05,
    scenario: str = "single",
    misalignment_deg: float = 2.0,
    mean_photon_number: float = 0.1,
    distance_km: float = 50.0,
    attenuation_db_per_km: float = 0.2,
    detector_efficiency: float = 0.15,
    dark_count_rate: float = 5e-6,
    seed: int | None = 12345,
):
    """
    BB84 with:

        quantum channel noise
        +
        alignment
        +
        WCP channel loss
        +
        detector efficiency
        +
        dark counts

    Important:
        A transmission with no detector event is discarded
        before basis sifting.

    Returns both detection statistics and QBER statistics.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Alice/Bob random choices
    # ---------------------------------------------------------

    alice_bits = rng.integers(
        0,
        2,
        size=n,
    )

    alice_bases = rng.integers(
        0,
        2,
        size=n,
    )

    bob_bases = rng.integers(
        0,
        2,
        size=n,
    )

    # ---------------------------------------------------------
    # Build circuits
    # ---------------------------------------------------------

    circuits = []

    for i in range(n):

        qc = build_bb84_quantum_circuit(
            alice_bit=int(alice_bits[i]),
            alice_basis=int(alice_bases[i]),
            bob_basis=int(bob_bases[i]),
            p=p,
            scenario=scenario,
            misalignment_deg=misalignment_deg,
        )

        circuits.append(qc)

    # ---------------------------------------------------------
    # Run quantum simulation
    # ---------------------------------------------------------

    backend = AerSimulator()

    result = backend.run(
        circuits,
        shots=1,
        seed_simulator=seed,
    ).result()

    quantum_bits = np.empty(
        n,
        dtype=int,
    )

    for i in range(n):

        counts = result.get_counts(i)

        outcome = next(iter(counts))

        quantum_bits[i] = int(outcome)

    # ---------------------------------------------------------
    # Detector layer
    # ---------------------------------------------------------

    detected_bits = np.full(
        n,
        -1,
        dtype=int,
    )

    detected_mask = np.zeros(
        n,
        dtype=bool,
    )

    signal_mask = np.zeros(
        n,
        dtype=bool,
    )

    dark_mask = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(n):

        detected_bit, event = detect_bit(
            ideal_bit=int(quantum_bits[i]),
            mean_photon_number=mean_photon_number,
            distance_km=distance_km,
            attenuation_db_per_km=attenuation_db_per_km,
            detector_efficiency=detector_efficiency,
            dark_count_rate=dark_count_rate,
            rng=rng,
        )

        if event == "signal":

            detected_bits[i] = detected_bit
            detected_mask[i] = True
            signal_mask[i] = True

        elif event == "dark":

            detected_bits[i] = detected_bit
            detected_mask[i] = True
            dark_mask[i] = True

    # ---------------------------------------------------------
    # Detection filtering
    # ---------------------------------------------------------

    detected_indices = detected_mask

    detected_alice_bits = (
        alice_bits[detected_indices]
    )

    detected_bases_alice = (
        alice_bases[detected_indices]
    )

    detected_bases_bob = (
        bob_bases[detected_indices]
    )

    detected_bob_bits = (
        detected_bits[detected_indices]
    )

    # ---------------------------------------------------------
    # Basis sifting
    # ---------------------------------------------------------

    sift_mask = (
        detected_bases_alice
        == detected_bases_bob
    )

    alice_sifted = (
        detected_alice_bits[sift_mask]
    )

    bob_sifted = (
        detected_bob_bits[sift_mask]
    )

    # ---------------------------------------------------------
    # QBER
    # ---------------------------------------------------------

    if len(alice_sifted) == 0:
        qber = 0.0
    else:
        qber = float(
            np.mean(
                alice_sifted
                != bob_sifted
            )
        )

    # ---------------------------------------------------------
    # Dark events among sifted bits
    # ---------------------------------------------------------

    detected_event_types = np.full(
        n,
        "",
        dtype=object,
    )

    detected_event_types[
        signal_mask
    ] = "signal"

    detected_event_types[
        dark_mask
    ] = "dark"

    detected_event_types = (
        detected_event_types[
            detected_indices
        ]
    )

    sifted_event_types = (
        detected_event_types[
            sift_mask
        ]
    )

    sifted_dark_count = int(
        np.sum(
            sifted_event_types == "dark"
        )
    )

    sifted_signal_count = int(
        np.sum(
            sifted_event_types == "signal"
        )
    )

    # ---------------------------------------------------------
    # Return complete statistics
    # ---------------------------------------------------------

    return {
        "qber": qber,

        "total_bits": n,

        "detected_bits": int(
            np.sum(detected_mask)
        ),

        "signal_detections": int(
            np.sum(signal_mask)
        ),

        "dark_detections": int(
            np.sum(dark_mask)
        ),

        "sifted_length": len(
            alice_sifted
        ),

        "sifted_signal_bits":
            sifted_signal_count,

        "sifted_dark_bits":
            sifted_dark_count,

        "alice_sifted":
            alice_sifted,

        "bob_sifted":
            bob_sifted,

        "alice_bits":
            alice_bits,

        "alice_bases":
            alice_bases,

        "bob_bases":
            bob_bases,

        "quantum_bits":
            quantum_bits,

        "detected_bits_array":
            detected_bits,

        "detected_mask":
            detected_mask,

        "signal_mask":
            signal_mask,

        "dark_mask":
            dark_mask,
    }
