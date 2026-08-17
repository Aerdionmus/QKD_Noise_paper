import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_aer.noise import amplitude_damping_error


SHOTS = 100_000

P_VALUES = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10
]


def paper_depolarizing_error(p):
    """
    Paper Eq. (3):

        P(I) = 1-p
        P(X) = P(Y) = P(Z) = p/3
    """

    return pauli_error([
        ("I", 1.0 - p),
        ("X", p / 3.0),
        ("Y", p / 3.0),
        ("Z", p / 3.0),
    ])


def paper_dephasing_error(p):
    """
    Paper Eq. (4):

        P(I) = 1-p
        P(Z) = p
    """

    return pauli_error([
        ("I", 1.0 - p),
        ("Z", p),
    ])


def paper_amplitude_damping_error(p):
    """
    Paper Eq. (5)-(6), with gamma=p.
    """

    return amplitude_damping_error(p)


def make_noise_model(p, scenario):
    """
    Construct the literal Track-A channel cascade.

    The important point is that the noise is attached to
    a gate that actually exists in the circuit.

    We use an explicit identity gate as the channel location.
    """

    noise = NoiseModel()

    dep = paper_depolarizing_error(p)

    if scenario == "single":

        noise.add_all_qubit_quantum_error(
            dep,
            ["id"]
        )

    elif scenario == "dual":

        deph = paper_dephasing_error(p)

        composite = dep.compose(deph)

        noise.add_all_qubit_quantum_error(
            composite,
            ["id"]
        )

    elif scenario == "triple":

        deph = paper_dephasing_error(p)
        amp = paper_amplitude_damping_error(p)

        composite = dep.compose(deph).compose(amp)

        noise.add_all_qubit_quantum_error(
            composite,
            ["id"]
        )

    else:
        raise ValueError(
            "scenario must be single, dual, or triple"
        )

    return noise


def make_circuit(initial_state):
    """
    Prepare a BB84 state, apply an explicit identity gate
    so that Aer's noise model has a gate location, then measure.

    The identity gate represents the channel location.
    """

    qc = QuantumCircuit(1, 1)

    if initial_state == "|0>":
        pass

    elif initial_state == "|1>":
        qc.x(0)

    elif initial_state == "|+>":
        qc.h(0)

    elif initial_state == "|->":
        qc.x(0)
        qc.h(0)

    else:
        raise ValueError(
            f"Unknown initial state: {initial_state}"
        )

    # Explicit channel location.
    qc.id(0)

    qc.measure(0, 0)

    return qc


def aer_probability_one(p, scenario, initial_state):
    """
    Measure P(1) after the literal paper channel cascade.
    """

    simulator = AerSimulator()

    noise = make_noise_model(
        p,
        scenario
    )

    circuit = make_circuit(
        initial_state
    )

    circuit = transpile(
        circuit,
        simulator
    )

    result = simulator.run(
        circuit,
        noise_model=noise,
        shots=SHOTS
    ).result()

    counts = result.get_counts()

    return counts.get("1", 0) / SHOTS


def analytical_z_error(p, scenario):
    """
    Analytical Z-basis error for the |0> input.

    This is NOT the four-state BB84 average.

    For |0>, the error probability is simply P(1).
    """

    if scenario == "single":

        # X and Y each cause a bit flip.
        return 2.0 * p / 3.0

    if scenario == "dual":

        # Depolarizing followed by phase flip.
        #
        # Z phase flips do not change computational-basis
        # populations, so the Z-basis error remains the
        # depolarizing bit-flip probability.
        return 2.0 * p / 3.0

    if scenario == "triple":

        # After depolarizing + dephasing, the probability of
        # |1> is 2p/3. Amplitude damping then maps |1> -> |0>
        # with probability p.
        #
        # Therefore:
        #
        # P(1)_out = P(1)_before_AD * (1-p)
        return (2.0 * p / 3.0) * (1.0 - p)

    raise ValueError(
        "scenario must be single, dual, or triple"
    )


def exact_four_state_qber(p, scenario):
    """
    Four-state BB84 reference.

    This independently evaluates the literal Kraus channels
    from src/qkd_noise/channels.py.

    This function is used only as a cross-check against the
    Aer implementation.
    """

    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from qkd_noise.channels import bb84_average_qber

    return bb84_average_qber(
        p,
        scenario=scenario
    )


print("Track-A full Aer convergence audit")
print("===================================")
print()
print(f"Shots per point : {SHOTS:,}")
print()

for p in P_VALUES:

    print(f"p = {p:.2f}")

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        analytical_z = analytical_z_error(
            p,
            scenario
        )

        aer_z = aer_probability_one(
            p,
            scenario,
            "|0>"
        )

        difference = aer_z - analytical_z

        print(
            f"  {scenario:7s}"
            f" Z-reference={analytical_z:.6f}"
            f" Aer={aer_z:.6f}"
            f" difference={difference:+.6f}"
        )

    print()

print("Four-state BB84 reference cross-check")
print("=====================================")
print()

for p in [0.01, 0.05, 0.10]:

    print(f"p = {p:.2f}")

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        qber = exact_four_state_qber(
            p,
            scenario
        )

        print(
            f"  {scenario:7s}"
            f" four-state QBER={qber:.6f}"
        )

    print()

print("Audit complete.")