import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.protocols.e91 import (
    E91_QBER_THRESHOLD,
    E91_VISIBILITY,
    apply_noise,
    chsh_value,
    simulate_e91,
    werner_state,
)


P_VALUES = np.arange(
    0.01,
    0.101,
    0.01,
)

SCENARIOS = [
    "single",
    "dual",
    "triple",
]

NUM_PAIRS = 1000
TRIALS = 50
SEED = 20260817

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_e91"
)

RAW_FILE = (
    RESULTS_DIR
    / "e91_final_trials.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "e91_final_summary.csv"
)


def exact_reference(
    p,
    scenario,
):
    """
    Calculate the exact density-matrix reference
    for the requested noise configuration.
    """

    rho = werner_state(
        E91_VISIBILITY
    )

    noisy = apply_noise(
        rho,
        p,
        scenario,
    )

    return {
        "exact_qber": None,
        "exact_chsh": chsh_value(noisy),
    }


def run_experiment():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_rows = []
    summary_rows = []

    total_runs = (
        len(P_VALUES)
        * len(SCENARIOS)
        * TRIALS
    )

    run_number = 0

    print(
        "FINAL E91 reproduction experiment"
    )
    print("=" * 42)
    print()

    print(
        f"p values       : "
        f"{list(P_VALUES)}"
    )

    print(
        f"scenarios      : "
        f"{SCENARIOS}"
    )

    print(
        f"pairs/trial    : "
        f"{NUM_PAIRS}"
    )

    print(
        f"trials         : "
        f"{TRIALS}"
    )

    print(
        f"total runs     : "
        f"{total_runs}"
    )

    print(
        f"visibility     : "
        f"{E91_VISIBILITY}"
    )

    print(
        f"QBER threshold : "
        f"{E91_QBER_THRESHOLD}"
    )

    print(
        f"seed           : "
        f"{SEED}"
    )

    print()

    for scenario in SCENARIOS:

        for p in P_VALUES:

            p = float(p)

            reference = exact_reference(
                p,
                scenario,
            )

            exact_chsh = reference[
                "exact_chsh"
            ]

            qber_values = []
            chsh_values = []
            sifted_values = []
            secure_values = []

            for trial in range(
                1,
                TRIALS + 1,
            ):

                run_number += 1

                trial_seed = (
                    SEED
                    + run_number
                )

                result = simulate_e91(
                    num_pairs=NUM_PAIRS,
                    p=p,
                    scenario=scenario,
                    visibility=E91_VISIBILITY,
                    seed=trial_seed,
                )

                qber = float(
                    result.qber
                )

                chsh = float(
                    result.chsh_s
                )

                sifted = int(
                    result.sifted_length
                )

                secure = bool(
                    result.secure
                )

                qber_values.append(
                    qber
                )

                chsh_values.append(
                    chsh
                )

                sifted_values.append(
                    sifted
                )

                secure_values.append(
                    secure
                )

                raw_rows.append(
                    {
                        "p": p,
                        "scenario": scenario,
                        "trial": trial,
                        "num_pairs": NUM_PAIRS,
                        "seed": trial_seed,
                        "qber": qber,
                        "chsh_s": chsh,
                        "sifted_length": sifted,
                        "secure": secure,
                    }
                )

                if (
                    trial == 1
                    or trial == TRIALS
                ):
                    print(
                        f"{run_number:4d}/"
                        f"{total_runs} | "
                        f"{scenario:<7} | "
                        f"p={p:.2f} | "
                        f"trial="
                        f"{trial:02d} | "
                        f"QBER="
                        f"{qber:.6f} | "
                        f"S="
                        f"{chsh:.6f} | "
                        f"sifted="
                        f"{sifted}"
                    )

            qber_array = np.array(
                qber_values
            )

            chsh_array = np.array(
                chsh_values
            )

            sifted_array = np.array(
                sifted_values
            )

            mean_qber = float(
                np.mean(qber_array)
            )

            std_qber = float(
                np.std(
                    qber_array,
                    ddof=1,
                )
            )

            mean_chsh = float(
                np.mean(chsh_array)
            )

            std_chsh = float(
                np.std(
                    chsh_array,
                    ddof=1,
                )
            )

            mean_sifted = float(
                np.mean(sifted_array)
            )

            sem_qber = (
                std_qber
                / np.sqrt(TRIALS)
            )

            sem_chsh = (
                std_chsh
                / np.sqrt(TRIALS)
            )

            ci95_qber_low = (
                mean_qber
                - 1.96 * sem_qber
            )

            ci95_qber_high = (
                mean_qber
                + 1.96 * sem_qber
            )

            ci95_chsh_low = (
                mean_chsh
                - 1.96 * sem_chsh
            )

            ci95_chsh_high = (
                mean_chsh
                + 1.96 * sem_chsh
            )

            chsh_difference = (
                mean_chsh
                - exact_chsh
            )

            summary_rows.append(
                {
                    "p": p,
                    "scenario": scenario,
                    "trials": TRIALS,
                    "num_pairs_per_trial": NUM_PAIRS,
                    "mean_qber": mean_qber,
                    "std_qber": std_qber,
                    "ci95_qber_low": ci95_qber_low,
                    "ci95_qber_high": ci95_qber_high,
                    "mean_chsh": mean_chsh,
                    "std_chsh": std_chsh,
                    "ci95_chsh_low": ci95_chsh_low,
                    "ci95_chsh_high": ci95_chsh_high,
                    "exact_chsh": exact_chsh,
                    "difference_from_exact_chsh": chsh_difference,
                    "mean_sifted_length": mean_sifted,
                    "secure_fraction": float(
                        np.mean(
                            secure_values
                        )
                    ),
                }
            )

    print()
    print("Experiment complete.")

    with RAW_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=raw_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(raw_rows)

    with SUMMARY_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"Raw results     : {RAW_FILE}"
    )

    print(
        f"Summary results : {SUMMARY_FILE}"
    )


def main():
    run_experiment()


if __name__ == "__main__":
    main()