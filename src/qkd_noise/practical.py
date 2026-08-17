import math

from qiskit import QuantumCircuit


def alignment_angle_radians(
    misalignment_deg: float,
) -> float:
    """
    Convert the specified misalignment angle from degrees
    to radians.
    """
    return math.radians(misalignment_deg)


def theoretical_alignment_qber(
    misalignment_deg: float,
) -> float:
    """
    Paper's analytical alignment-error expression:

        Q_align = sin^2(epsilon)

    where epsilon is given in degrees.
    """

    epsilon = alignment_angle_radians(
        misalignment_deg
    )

    return math.sin(epsilon) ** 2


def apply_alignment_error(
    qc: QuantumCircuit,
    misalignment_deg: float,
):
    """
    Apply a physical basis-independent misalignment
    before Bob's measurement.

    We use Ry(2*epsilon).

    Since a Ry(theta) rotation gives a transition
    probability sin^2(theta/2), choosing theta=2*epsilon
    gives:

        P(error) = sin^2(epsilon)

    matching the paper's analytical Q_align expression.
    """

    epsilon = alignment_angle_radians(
        misalignment_deg
    )

    qc.ry(
        2.0 * epsilon,
        0,
    )
