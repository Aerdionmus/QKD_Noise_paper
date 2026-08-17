import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkd_noise.detector import detect_bit


def main():

    n = 1_000_000

    mu = 0.1
    distance = 50.0
    attenuation = 0.2
    efficiency = 0.15
    dark_rate = 5e-6

    rng = np.random.default_rng(12345)

    signal_count = 0
    dark_count = 0
    none_count = 0
    errors = 0
    detected_count = 0

    for _ in range(n):

        # Equal mixture of logical 0 and 1.
        ideal_bit = int(
            rng.integers(0, 2)
        )

        detected_bit, event = detect_bit(
            ideal_bit=ideal_bit,
            mean_photon_number=mu,
            distance_km=distance,
            attenuation_db_per_km=attenuation,
            detector_efficiency=efficiency,
            dark_count_rate=dark_rate,
            rng=rng,
        )

        if event == "signal":
            signal_count += 1
            detected_count += 1

        elif event == "dark":
            dark_count += 1
            detected_count += 1

        else:
            none_count += 1

        if detected_bit is not None:
            if detected_bit != ideal_bit:
                errors += 1

    print("Detector Monte-Carlo diagnostic")
    print("===============================")
    print(f"Trials             : {n}")
    print(f"Signal detections  : {signal_count}")
    print(f"Dark detections    : {dark_count}")
    print(f"No detection       : {none_count}")
    print(f"Total detections   : {detected_count}")
    print()

    print(
        f"Detection rate     : "
        f"{detected_count / n:.8f}"
    )

    print(
        f"Dark fraction      : "
        f"{dark_count / detected_count:.8f}"
    )

    print(
        f"Conditional QBER   : "
        f"{errors / detected_count:.8f}"
    )


if __name__ == "__main__":
    main()
