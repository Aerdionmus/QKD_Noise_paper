from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
)


GAMMA = 0.05
SHOTS = 1_000_000


def make_one_circuit():
    """
    Prepare |1>, apply amplitude damping, measure Z.

        |0> -- X -- channel -- M
    """

    qc = QuantumCircuit(1, 1)

    qc.x(0)
    qc.id(0)
    qc.measure(0, 0)

    return qc


def make_zero_circuit():
    """
    Prepare |0>, apply amplitude damping, measure Z.
    """

    qc = QuantumCircuit(1, 1)

    qc.id(0)
    qc.measure(0, 0)

    return qc


def run(circuit):
    noise = NoiseModel()

    error = amplitude_damping_error(
        GAMMA
    )

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

    p_zero = (
        counts.get("0", 0)
        / SHOTS
    )

    p_one = (
        counts.get("1", 0)
        / SHOTS
    )

    return counts, p_zero, p_one


def main():

    print("Amplitude-damping channel comparison")
    print("=====================================")
    print()

    print(
        f"gamma = {GAMMA}"
    )

    print(
        f"shots = {SHOTS:,}"
    )

    print()

    # ========================================================
    # |1>
    # ========================================================

    counts, p_zero, p_one = run(
        make_one_circuit()
    )

    expected_zero = GAMMA
    expected_one = 1.0 - GAMMA

    print("|1> INPUT")
    print("---------")

    print(
        f"Expected P(0) : "
        f"{expected_zero:.8f}"
    )

    print(
        f"Aer P(0)      : "
        f"{p_zero:.8f}"
    )

    print(
        f"Difference     : "
        f"{p_zero - expected_zero:+.8f}"
    )

    print(
        f"Expected P(1) : "
        f"{expected_one:.8f}"
    )

    print(
        f"Aer P(1)      : "
        f"{p_one:.8f}"
    )

    print(
        f"Counts        : "
        f"{counts}"
    )

    print()

    # ========================================================
    # |0>
    # ========================================================

    counts, p_zero, p_one = run(
        make_zero_circuit()
    )

    print("|0> INPUT")
    print("---------")

    print(
        "Expected P(0) : 1.00000000"
    )

    print(
        f"Aer P(0)      : "
        f"{p_zero:.8f}"
    )

    print(
        "Expected P(1) : 0.00000000"
    )

    print(
        f"Aer P(1)      : "
        f"{p_one:.8f}"
    )

    print(
        f"Counts        : "
        f"{counts}"
    )

    print()

    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
