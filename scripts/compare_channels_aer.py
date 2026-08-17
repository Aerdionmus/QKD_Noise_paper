from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


P = 0.05
SHOTS = 1_000_000


def make_z0_circuit():
    qc = QuantumCircuit(1, 1)

    # Explicit quantum operation where the channel is applied.
    qc.id(0)

    qc.measure(0, 0)

    return qc


def paper_depolarizing_error(p):
    """
    Paper-style Pauli depolarizing channel:

        P(I) = 1 - p
        P(X) = p/3
        P(Y) = p/3
        P(Z) = p/3

    This corresponds to:

        D(rho)
          = (1-p)rho
            + p/3 [
                X rho X
                + Y rho Y
                + Z rho Z
              ]
    """

    return pauli_error(
        [
            ("I", 1.0 - p),
            ("X", p / 3.0),
            ("Y", p / 3.0),
            ("Z", p / 3.0),
        ]
    )


def qiskit_depolarizing_error(p):
    """
    Qiskit's built-in depolarizing channel.

    This is included only as a reference showing the
    parameterization difference.
    """

    from qiskit_aer.noise import depolarizing_error

    return depolarizing_error(
        p,
        1,
    )


def run(error):
    noise = NoiseModel()

    noise.add_all_qubit_quantum_error(
        error,
        ["id"],
    )

    simulator = AerSimulator(
        noise_model=noise
    )

    qc = make_z0_circuit()

    result = simulator.run(
        qc,
        shots=SHOTS,
    ).result()

    counts = result.get_counts()

    p_one = (
        counts.get("1", 0)
        / SHOTS
    )

    return counts, p_one


def main():

    print(
        "Depolarizing parameterization comparison"
    )
    print(
        "========================================="
    )
    print()

    print(
        f"p     = {P}"
    )

    print(
        f"shots = {SHOTS:,}"
    )

    print()

    # --------------------------------------------------------
    # Paper-style channel
    # --------------------------------------------------------

    paper_error = paper_depolarizing_error(
        P
    )

    counts, measured = run(
        paper_error
    )

    theoretical = (
        2.0 * P / 3.0
    )

    print("PAPER-STYLE PAULI CHANNEL")
    print("-------------------------")

    print(
        f"Theoretical P(1) : "
        f"{theoretical:.8f}"
    )

    print(
        f"Aer P(1)         : "
        f"{measured:.8f}"
    )

    print(
        f"Difference        : "
        f"{measured - theoretical:+.8f}"
    )

    print(
        f"Counts            : "
        f"{counts}"
    )

    print()

    # --------------------------------------------------------
    # Qiskit built-in channel
    # --------------------------------------------------------

    qiskit_error = (
        qiskit_depolarizing_error(P)
    )

    counts, measured = run(
        qiskit_error
    )

    qiskit_theoretical = (
        P / 2.0
    )

    print("QISKIT BUILT-IN CHANNEL")
    print("-----------------------")

    print(
        f"Theoretical P(1) : "
        f"{qiskit_theoretical:.8f}"
    )

    print(
        f"Aer P(1)         : "
        f"{measured:.8f}"
    )

    print(
        f"Difference        : "
        f"{measured - qiskit_theoretical:+.8f}"
    )

    print(
        f"Counts            : "
        f"{counts}"
    )

    print()

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    print("CONCLUSION")
    print("==========")

    print(
        "The two channels use different parameterizations."
    )

    print()

    print(
        "Paper-style:"
    )

    print(
        "    P(X)=P(Y)=P(Z)=p/3"
    )

    print(
        "    P(I)=1-p"
    )

    print()

    print(
        "Qiskit built-in:"
    )

    print(
        "    depolarizing_error(p, 1)"
    )

    print(
        "    is parameterized differently."
    )


if __name__ == "__main__":
    main()