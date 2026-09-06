import os
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer


# ============================================================
# OUTPUT
# ============================================================

OUT = "results/paper_figures"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# DISPLAY-ONLY GATES
#
# These gates are schematic representations for the paper
# figures. The actual quantum channels are implemented in the
# project noise engine.
# ============================================================

def depolarizing_gate():
    return Gate(
        name=r"$\mathcal{K}_{\mathrm{dep}}$",
        num_qubits=1,
        params=[],
    )


def dephasing_gate():
    return Gate(
        name=r"$\mathcal{K}_{\mathrm{deph}}$",
        num_qubits=1,
        params=[],
    )


def amplitude_damping_gate():
    return Gate(
        name=r"$\mathcal{K}_{\mathrm{AD}}$",
        num_qubits=1,
        params=[],
    )


def add_composite_noise(qc, qubit):
    """
    Schematic representation of the implemented composite
    channel:

        N_AD o N_deph o N_dep
    """

    qc.append(depolarizing_gate(), [qubit])
    qc.append(dephasing_gate(), [qubit])
    qc.append(amplitude_damping_gate(), [qubit])


# ============================================================
# BB84
# ============================================================

def make_bb84():

    qc = QuantumCircuit(1, 1)

    # Representative BB84 preparation:
    # |+> = H|0>
    qc.h(0)

    # Composite noise channel
    add_composite_noise(qc, 0)

    # Bob measures in the X basis
    qc.h(0)

    # Detection
    qc.measure(0, 0)

    return qc


# ============================================================
# B92
# ============================================================

def b92_preparation_gate():
    """
    Display-only representation of the B92 preparation stage.

    The protocol uses the two non-orthogonal states:
        |0> and |+>
    """

    return Gate(
        name=r"$|0\rangle/|+\rangle$",
        num_qubits=1,
        params=[],
    )


def b92_measurement_gate():
    """
    Display-only representation of the B92 measurement stage.

    The detailed conclusive/inconclusive POVM mapping is kept
    schematic because it is not fully specified in the source
    material used for this project.
    """

    return Gate(
        name=r"$\mathrm{B92\ POVM}$",
        num_qubits=1,
        params=[],
    )


def make_b92():

    qc = QuantumCircuit(1, 1)

    # B92 preparation:
    # Alice selects one of the two non-orthogonal states
    # |0> and |+>.
    qc.append(b92_preparation_gate(), [0])

    # Composite noise channel
    add_composite_noise(qc, 0)

    # B92 measurement stage
    qc.append(b92_measurement_gate(), [0])

    # Detection outcome
    qc.measure(0, 0)

    return qc


# ============================================================
# E91
# ============================================================

def make_e91():

    qc = QuantumCircuit(2, 2)

    # --------------------------------------------------------
    # Entangled-pair preparation
    # --------------------------------------------------------

    qc.h(0)
    qc.cx(0, 1)

    # --------------------------------------------------------
    # Composite noise on Bob's qubit
    # --------------------------------------------------------

    add_composite_noise(qc, 1)

    # --------------------------------------------------------
    # Representative measurement settings
    # --------------------------------------------------------

    qc.ry(np.pi / 4, 0)
    qc.ry(-np.pi / 8, 1)

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# ============================================================
# SAVE CIRCUITS
# ============================================================

def save_circuit(qc, filename):

    circuit_drawer(
        qc,
        output="mpl",
        filename=filename,
        fold=-1,
        scale=1.2,
    )

    print(f"Created: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Generating publication circuit diagrams")
    print("========================================")
    print()

    print("Generating BB84...")
    save_circuit(
        make_bb84(),
        f"{OUT}/bb84_qiskit_circuit.png",
    )

    print("Generating B92...")
    save_circuit(
        make_b92(),
        f"{OUT}/b92_qiskit_circuit.png",
    )

    print("Generating E91...")
    save_circuit(
        make_e91(),
        f"{OUT}/e91_qiskit_circuit.png",
    )

    print()
    print("Complete.")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()