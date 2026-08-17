import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ============================================================
# PAPER PARAMETERS
# ============================================================

P_VALUES = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
]

MISALIGNMENT_DEG = 2.0
DARK_COUNT_RATE = 5e-6
MEAN_PHOTON_NUMBER = 0.1
DISTANCE_KM = 50.0
ATTENUATION_DB_PER_KM = 0.2
DETECTOR_EFFICIENCY = 0.15

Q0_REPORTED = 0.006


# ============================================================
# EQUATION (ALIGNMENT)
# ============================================================

def q_align(misalignment_deg):
    """
    Q_align = sin^2(theta)

    IMPORTANT:
    This implementation uses the equation currently
    documented in our reproduction notes.
    """

    theta = math.radians(misalignment_deg)

    return math.sin(theta) ** 2


# ============================================================
# EQUATION (PNS)
# ============================================================

def delta_pns_literal(mu):
    """
    Literal PNS expression currently recorded from the paper.

    Do NOT replace or "correct" this equation here.
    """

    return 1.0 - math.exp(-mu)


# ============================================================
# DARK-COUNT EXPRESSION
# ============================================================

def dark_count_literal(
    dark_count_rate,
    detector_efficiency,
    mean_photon_number,
    distance_km,
    attenuation_db_per_km,
):
    """
    Literal dark-count calculation used in the current
    reproduction notes.

    This function is intentionally isolated because the
    manuscript's expression requires separate auditing.

    We will compare its numerical behavior against the
    physical detector model later.
    """

    transmittance = (
        10
        ** (
            -attenuation_db_per_km
            * distance_km
            / 10.0
        )
    )

    signal_probability = (
        1.0
        - math.exp(
            -mean_photon_number
            * transmittance
            * detector_efficiency
        )
    )

    if signal_probability == 0:
        return float("inf")

    return dark_count_rate / signal_probability


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("Paper-literal practical QBER audit")
    print("===================================")
    print()

    print("Parameters")
    print("----------")
    print(f"Misalignment       : {MISALIGNMENT_DEG} degrees")
    print(f"Dark count rate    : {DARK_COUNT_RATE:.6e}")
    print(f"Mean photon number : {MEAN_PHOTON_NUMBER}")
    print(f"Distance           : {DISTANCE_KM} km")
    print(f"Attenuation        : {ATTENUATION_DB_PER_KM} dB/km")
    print(f"Detector efficiency: {DETECTOR_EFFICIENCY}")
    print(f"Q0 reported        : {Q0_REPORTED}")
    print()

    q_alignment = q_align(
        MISALIGNMENT_DEG
    )

    pns = delta_pns_literal(
        MEAN_PHOTON_NUMBER
    )

    dark = dark_count_literal(
        DARK_COUNT_RATE,
        DETECTOR_EFFICIENCY,
        MEAN_PHOTON_NUMBER,
        DISTANCE_KM,
        ATTENUATION_DB_PER_KM,
    )

    print("Fixed analytical quantities")
    print("---------------------------")
    print(
        f"Q_align            : "
        f"{q_alignment:.8f}"
    )

    print(
        f"Delta_PNS literal  : "
        f"{pns:.8e}"
    )

    print(
        f"Dark-count literal : "
        f"{dark:.8e}"
    )

    print()

    print("Paper parameter sweep")
    print("---------------------")

    print(
        "p       Q_align      Delta_PNS      "
        "Dark-term"
    )


    for p in P_VALUES:

        pns_p = delta_pns_literal(
            p
        )

        print(
            f"{p:.2f}    "
            f"{q_alignment:.8f}   "
            f"{pns_p:.8e}   "
            f"{dark:.8e}"
        )


if __name__ == "__main__":
    main()
