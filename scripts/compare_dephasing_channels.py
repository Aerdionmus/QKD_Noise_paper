from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    pauli_error,
    phase_damping_error,
)


P = 0.05
SHOTS = 1_000_000


def make_x_basis_circuit():
    """
    Prepare |+>, apply the channel, then measure in
    the X basis.

        |0> -- H -- channel -- H -- M
    """

    qc = QuantumCircuit(1, 1)

    # Prepare |+>
    qc.h(0)

    # Channel is attached to this explicit identity gate.
    qc.id(0)

    # Return to computational basis for X-basis measurement.
    qc.h(0)

    qc.measure(0, 0)

    return qc


def phase_flip_error(p):
    """
    Phase-flip channel:

        P(I) = 1-p
        P(Z) = p

    Equivalent to:

        rho' = (1-p)rho + p Z rho Z
    """

    return pauli_error(
        [
            ("I", 1.0 - p),
            ("Z", p),
        ]
    )


def run(circuit, error):
    noise = NoiseModel()

    noise.add_all_qubit_quantum_error(
        error,
        ["id"],
    )

    simulator = AerSimulator(
        noise_model=noise
    )

    result = simulator.run(
        circuit,
        shots=SHOTS,
    ).result()

    counts = result.get_counts()

    p_error = (
        counts.get("1", 0)
        / SHOTS
    )

    return counts, p_error


def main():

    print("Phase-noise X-basis comparison")
    print("==============================")
    print()

    print(f"p     = {P}")
    print(f"shots = {SHOTS:,}")
    print()

    circuit = make_x_basis_circuit()

    # ========================================================
    # Phase flip
    # ========================================================

    error = phase_flip_error(P)

    counts, measured = run(
        circuit,
        error,
    )

    theoretical = P

    print("PHASE-FLIP")
    print("----------")

    print(
        f"Theoretical X-basis error : "
        f"{theoretical:.8f}"
    )

    print(
        f"Aer X-basis error         : "
        f"{measured:.8f}"
    )

    print(
        f"Difference                : "
        f"{measured - theoretical:+.8f}"
    )

    print(
        f"Counts                    : "
        f"{counts}"
    )

    print()

    # ========================================================
    # Phase damping
    # ========================================================

    error = phase_damping_error(
        P
    )

    counts, measured = run(
        circuit,
        error,
    )

    print("PHASE-DAMPING")
    print("-------------")

    print(
        f"Aer X-basis error         : "
        f"{measured:.8f}"
    )

    print(
        f"Counts                    : "
        f"{counts}"
    )

    print()

    print("INTERPRETATION")
    print("==============")

    print()

    print(
        "This experiment measures the coherence affected"
    )

    print(
        "by the phase-noise channel."
    )

    print()

    print(
        "The phase-flip and phase-damping conventions should"
    )

    print(
        "not be assumed equivalent."
    )

    print()

    print(
        "We will use this result to determine which convention"
    )

    print(
        "is appropriate for the Track-A implementation."
    )


if __name__ == "__main__":
    main()