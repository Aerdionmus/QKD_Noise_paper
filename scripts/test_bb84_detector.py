import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkd_noise.protocols.bb84_detector import (
    simulate_bb84_detector,
)


def main():

    n = 100000
    p = 0.05

    print("BB84 practical detector integration")
    print("====================================")
    print(f"Transmissions : {n}")
    print(f"p             : {p}")
    print()

    for scenario in (
        "single",
        "dual",
        "triple",
    ):

        result = simulate_bb84_detector(
            n=n,
            p=p,
            scenario=scenario,
            misalignment_deg=2.0,
            mean_photon_number=0.1,
            distance_km=50.0,
            attenuation_db_per_km=0.2,
            detector_efficiency=0.15,
            dark_count_rate=5e-6,
            seed=20260817,
        )

        print(scenario.upper())

        print(
            f"  QBER              : "
            f"{result['qber']:.6f}"
        )

        print(
            f"  Detected bits     : "
            f"{result['detected_bits']}"
        )

        print(
            f"  Signal detections : "
            f"{result['signal_detections']}"
        )

        print(
            f"  Dark detections   : "
            f"{result['dark_detections']}"
        )

        print(
            f"  Sifted bits       : "
            f"{result['sifted_length']}"
        )

        print(
            f"  Sifted signal     : "
            f"{result['sifted_signal_bits']}"
        )

        print(
            f"  Sifted dark       : "
            f"{result['sifted_dark_bits']}"
        )

        print()


if __name__ == "__main__":
    main()
