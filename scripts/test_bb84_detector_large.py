import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkd_noise.protocols.bb84_detector import (
    simulate_bb84_detector,
)


def run_scenario(
    scenario: str,
    total_transmissions: int,
    chunk_size: int,
    p: float,
    seed: int,
):
    """
    Run the practical detector simulation in chunks.

    QBER is calculated from the accumulated sifted bits,
    not by averaging the individual chunk QBERs.
    """

    total_detected = 0
    total_signal = 0
    total_dark = 0
    total_sifted = 0
    total_sifted_signal = 0
    total_sifted_dark = 0

    total_errors = 0

    n_chunks = (
        total_transmissions // chunk_size
    )

    if total_transmissions % chunk_size != 0:
        raise ValueError(
            "total_transmissions must be divisible "
            "by chunk_size"
        )

    for chunk in range(n_chunks):

        chunk_seed = seed + chunk

        result = simulate_bb84_detector(
            n=chunk_size,
            p=p,
            scenario=scenario,
            misalignment_deg=2.0,
            mean_photon_number=0.1,
            distance_km=50.0,
            attenuation_db_per_km=0.2,
            detector_efficiency=0.15,
            dark_count_rate=5e-6,
            seed=chunk_seed,
        )

        total_detected += result[
            "detected_bits"
        ]

        total_signal += result[
            "signal_detections"
        ]

        total_dark += result[
            "dark_detections"
        ]

        total_sifted += result[
            "sifted_length"
        ]

        total_sifted_signal += result[
            "sifted_signal_bits"
        ]

        total_sifted_dark += result[
            "sifted_dark_bits"
        ]

        # Recover the number of errors from the
        # accumulated sifted arrays.
        total_errors += int(
            np.sum(
                result["alice_sifted"]
                != result["bob_sifted"]
            )
        )

        print(
            f"    chunk "
            f"{chunk + 1:02d}/{n_chunks} | "
            f"detected={result['detected_bits']:4d} | "
            f"sifted={result['sifted_length']:3d} | "
            f"QBER={result['qber']:.6f}"
        )

    if total_sifted == 0:
        qber = 0.0
    else:
        qber = (
            total_errors
            / total_sifted
        )

    return {
        "qber": qber,
        "detected": total_detected,
        "signal": total_signal,
        "dark": total_dark,
        "sifted": total_sifted,
        "sifted_signal": total_sifted_signal,
        "sifted_dark": total_sifted_dark,
        "errors": total_errors,
    }


def main():

    total_transmissions = 1_000_000
    chunk_size = 100_000
    p = 0.05

    print(
        "BB84 practical detector "
        "high-statistics test"
    )
    print(
        "======================================"
    )
    print(
        f"Total transmissions : "
        f"{total_transmissions:,}"
    )
    print(
        f"Chunk size          : "
        f"{chunk_size:,}"
    )
    print(
        f"p                   : {p}"
    )
    print(
        "Misalignment        : 2 degrees"
    )
    print(
        "Mean photon number  : 0.1"
    )
    print(
        "Distance            : 50 km"
    )
    print(
        "Attenuation         : 0.2 dB/km"
    )
    print(
        "Detector efficiency : 0.15"
    )
    print(
        "Dark count rate     : 5e-6"
    )
    print()

    for scenario in (
        "single",
        "dual",
        "triple",
    ):

        print(
            f"--- {scenario.upper()} ---"
        )

        result = run_scenario(
            scenario=scenario,
            total_transmissions=total_transmissions,
            chunk_size=chunk_size,
            p=p,
            seed=20260817,
        )

        print()
        print(
            f"  Total detected      : "
            f"{result['detected']}"
        )

        print(
            f"  Signal detections   : "
            f"{result['signal']}"
        )

        print(
            f"  Dark detections     : "
            f"{result['dark']}"
        )

        print(
            f"  Total sifted       : "
            f"{result['sifted']}"
        )

        print(
            f"  Sifted signal      : "
            f"{result['sifted_signal']}"
        )

        print(
            f"  Sifted dark        : "
            f"{result['sifted_dark']}"
        )

        print(
            f"  Errors             : "
            f"{result['errors']}"
        )

        print(
            f"  FINAL QBER         : "
            f"{result['qber']:.6f}"
        )

        print()


if __name__ == "__main__":
    main()
