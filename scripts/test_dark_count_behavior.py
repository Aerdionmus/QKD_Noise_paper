import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkd_noise.detector import detect_bit


def run_test(dark_rate, n=100000):

    rng = np.random.default_rng(12345)

    dark_events = 0
    dark_errors = 0

    for _ in range(n):

        ideal_bit = int(
            rng.integers(0, 2)
        )

        detected_bit, event = detect_bit(
            ideal_bit=ideal_bit,
            mean_photon_number=0.0,
            distance_km=50.0,
            attenuation_db_per_km=0.2,
            detector_efficiency=0.15,
            dark_count_rate=dark_rate,
            rng=rng,
        )

        if event == "dark":

            dark_events += 1

            if detected_bit != ideal_bit:
                dark_errors += 1

    if dark_events > 0:
        error_rate = (
            dark_errors / dark_events
        )
    else:
        error_rate = 0.0

    return dark_events, dark_errors, error_rate


def main():

    print("Dark-count behavior unit test")
    print("==============================")
    print()

    # No signal: every detection must come from a dark count.
    for dark_rate in (
        0.01,
        0.10,
        0.50,
    ):

        events, errors, error_rate = run_test(
            dark_rate
        )

        print(
            f"dark_rate={dark_rate:.2f}"
        )
        print(
            f"  dark events : {events}"
        )
        print(
            f"  errors      : {errors}"
        )
        print(
            f"  error rate  : {error_rate:.4f}"
        )
        print(
            f"  expected    : ~0.5000"
        )
        print()


if __name__ == "__main__":
    main()
