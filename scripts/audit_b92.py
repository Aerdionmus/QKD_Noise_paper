import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.protocols.b92 import (
    KET_0,
    KET_PLUS,
    density_matrix,
    apply_noise,
    z_measurement_probabilities,
    x_measurement_probabilities,
    simulate_b92,
)


P_VALUES = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10,
]

SCENARIOS = [
    "single",
    "dual",
    "triple",
]

TRANSMISSIONS = 1000
TRIALS = 50
SEED = 20260817


def exact_b92(p, scenario):
    """
    Exact B92 reference calculation.

    Alice:
        bit 0 -> |0>
        bit 1 -> |+>

    Bob:
        Z basis:
            outcome |1> is conclusive -> infer bit 1

        X basis:
            outcome |-> is conclusive -> infer bit 0

    QBER is conditioned on conclusive events.
    """

    states = {
        0: KET_0,
        1: KET_PLUS,
    }

    total_conclusive = 0.0
    total_errors = 0.0

    for alice_bit in (0, 1):

        rho = density_matrix(
            states[alice_bit]
        )

        rho = apply_noise(
            rho,
            p,
            scenario,
        )

        # Alice probability = 1/2
        # Bob basis probability = 1/2
        weight = 0.25

        # ---------------------------------------------------------
        # Bob Z basis
        # ---------------------------------------------------------

        _, p1 = z_measurement_probabilities(
            rho
        )

        total_conclusive += weight * p1

        # If Alice sent bit 0, Z outcome 1 is an error.
        if alice_bit == 0:
            total_errors += weight * p1

        # ---------------------------------------------------------
        # Bob X basis
        # ---------------------------------------------------------

        _, p_minus = x_measurement_probabilities(
            rho
        )

        total_conclusive += weight * p_minus

        # If Alice sent bit 1, X outcome - is an error.
        if alice_bit == 1:
            total_errors += weight * p_minus

    qber = (
        total_errors / total_conclusive
        if total_conclusive > 0.0
        else 0.0
    )

    return {
        "qber": qber,
        "conclusive_probability": total_conclusive,
        "error_probability": total_errors,
    }


def main():

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "results"
        / "b92_audit"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_FILE = (
        OUTPUT_DIR
        / "b92_audit_trials.csv"
    )

    SUMMARY_FILE = (
        OUTPUT_DIR
        / "b92_audit_summary.csv"
    )

    rng = np.random.default_rng(SEED)

    raw_rows = []
    summary_rows = []

    print("B92 STATISTICAL CONVERGENCE AUDIT")
    print("=================================")
    print()
    print(f"Transmissions per trial : {TRANSMISSIONS}")
    print(f"Trials per point        : {TRIALS}")
    print(
        f"Total runs              : "
        f"{len(P_VALUES) * len(SCENARIOS) * TRIALS}"
    )
    print(f"Seed                    : {SEED}")
    print()

    for scenario in SCENARIOS:

        for p in P_VALUES:

            exact = exact_b92(
                p,
                scenario,
            )

            qbers = []
            conclusive_fractions = []

            for trial in range(
                1,
                TRIALS + 1,
            ):

                trial_seed = int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                )

                result = simulate_b92(
                    n=TRANSMISSIONS,
                    p=p,
                    scenario=scenario,
                    seed=trial_seed,
                )

                qber = float(
                    result["qber"]
                )

                conclusive_fraction = float(
                    result["conclusive_fraction"]
                )

                qbers.append(qber)
                conclusive_fractions.append(
                    conclusive_fraction
                )

                raw_rows.append({
                    "p": p,
                    "scenario": scenario,
                    "trial": trial,
                    "seed": trial_seed,
                    "n_transmissions":
                        TRANSMISSIONS,
                    "conclusive_count":
                        result["conclusive_count"],
                    "conclusive_fraction":
                        conclusive_fraction,
                    "qber":
                        qber,
                    "exact_qber":
                        exact["qber"],
                    "difference_from_exact":
                        qber - exact["qber"],
                    "exact_conclusive_probability":
                        exact[
                            "conclusive_probability"
                        ],
                    "difference_conclusive":
                        (
                            conclusive_fraction
                            - exact[
                                "conclusive_probability"
                            ]
                        ),
                })

            qbers = np.asarray(
                qbers,
                dtype=float,
            )

            conclusive_fractions = np.asarray(
                conclusive_fractions,
                dtype=float,
            )

            mean_qber = float(
                np.mean(qbers)
            )

            std_qber = float(
                np.std(
                    qbers,
                    ddof=1,
                )
            )

            standard_error = (
                std_qber
                / np.sqrt(TRIALS)
            )

            ci_low = (
                mean_qber
                - 1.96 * standard_error
            )

            ci_high = (
                mean_qber
                + 1.96 * standard_error
            )

            mean_conclusive = float(
                np.mean(
                    conclusive_fractions
                )
            )

            summary_rows.append({
                "p": p,
                "scenario": scenario,
                "trials": TRIALS,
                "n_transmissions_per_trial":
                    TRANSMISSIONS,
                "mean_qber": mean_qber,
                "std_qber": std_qber,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "exact_qber": exact["qber"],
                "difference_from_exact":
                    mean_qber - exact["qber"],
                "ci_contains_exact":
                    (
                        ci_low
                        <= exact["qber"]
                        <= ci_high
                    ),
                "mean_conclusive_fraction":
                    mean_conclusive,
                "exact_conclusive_probability":
                    exact[
                        "conclusive_probability"
                    ],
                "difference_conclusive":
                    (
                        mean_conclusive
                        - exact[
                            "conclusive_probability"
                        ]
                    ),
            })

    raw_fields = [
        "p",
        "scenario",
        "trial",
        "seed",
        "n_transmissions",
        "conclusive_count",
        "conclusive_fraction",
        "qber",
        "exact_qber",
        "difference_from_exact",
        "exact_conclusive_probability",
        "difference_conclusive",
    ]

    summary_fields = [
        "p",
        "scenario",
        "trials",
        "n_transmissions_per_trial",
        "mean_qber",
        "std_qber",
        "ci95_low",
        "ci95_high",
        "exact_qber",
        "difference_from_exact",
        "ci_contains_exact",
        "mean_conclusive_fraction",
        "exact_conclusive_probability",
        "difference_conclusive",
    ]

    with RAW_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=raw_fields,
        )

        writer.writeheader()
        writer.writerows(raw_rows)

    with SUMMARY_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(
        "scenario   p       exact       mean"
        "        difference       95% CI"
    )
    print("-" * 90)

    for row in summary_rows:

        print(
            f"{row['scenario']:8s}"
            f" {row['p']:.2f}"
            f"    {row['exact_qber']:.6f}"
            f"    {row['mean_qber']:.6f}"
            f"    {row['difference_from_exact']:+.6f}"
            f"    [{row['ci95_low']:.6f}, "
            f"{row['ci95_high']:.6f}]"
        )

    containing = sum(
        row["ci_contains_exact"]
        for row in summary_rows
    )

    absolute_differences = [
        abs(
            row["difference_from_exact"]
        )
        for row in summary_rows
    ]

    print()
    print("SUMMARY")
    print("=======")
    print(
        "Maximum absolute difference : "
        f"{max(absolute_differences):.6f}"
    )
    print(
        "Mean absolute difference    : "
        f"{np.mean(absolute_differences):.6f}"
    )
    print(
        "95% CIs containing exact    : "
        f"{containing}/{len(summary_rows)}"
    )

    print()
    print("OUTPUT")
    print("======")
    print(f"Raw audit     : {RAW_FILE}")
    print(f"Summary audit : {SUMMARY_FILE}")

    print()
    print("Audit complete.")


if __name__ == "__main__":
    main()
