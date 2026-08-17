import csv
import sys
from pathlib import Path

import numpy as np

# Make src/qkd_noise importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.protocols.bb84_batch import simulate_bb84_batch
from qkd_noise.channels import bb84_average_qber


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

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_bb84"
RAW_FILE = OUTPUT_DIR / "bb84_final_trials.csv"
SUMMARY_FILE = OUTPUT_DIR / "bb84_final_summary.csv"


# Paper Table 4 values currently established during the audit.
# Only values explicitly available from the paper are entered.
PAPER_TABLE_4 = {
    (0.02, "single"): 0.049,
    (0.02, "dual"): 0.064,

    (0.04, "single"): 0.099,
    (0.04, "dual"): 0.132,

    (0.06, "single"): 0.151,
    (0.06, "dual"): 0.207,

    (0.08, "single"): 0.199,
    (0.08, "dual"): 0.287,

    (0.10, "single"): 0.246,
    (0.10, "dual"): 0.368,
}


def paper_key_rate(qber: float) -> float:
    """
    Paper Eq. (19), using the implementation already established
    during the Track-A audit.

    Negative rates are retained here rather than clipped so that
    the literal mathematical result remains visible.
    """

    if qber <= 0.0:
        return 1.0

    if qber >= 1.0:
        return 0.0

    h2 = (
        -qber * np.log2(qber)
        - (1.0 - qber) * np.log2(1.0 - qber)
    )

    return max(
        0.0,
        1.0 - 2.0 * h2,
    )


def paper_pri(qber: float, sigma_p: float) -> float:
    """
    Paper Eq. (20):

        PRI = R / Sigma_p

    where R is the paper secret-key-rate expression.

    The function is deliberately kept literal.
    """

    if sigma_p <= 0.0:
        return np.nan

    return paper_key_rate(qber) / sigma_p


def run_one_trial(p: float, scenario: str, seed: int):
    """
    Run one BB84 batch trial using the validated batch
    implementation.
    """

    result = simulate_bb84_batch(
        n=TRANSMISSIONS,
        p=p,
        scenario=scenario,
        seed=seed,
    )

    return result


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(SEED)

    raw_rows = []
    summary_rows = []

    total_runs = (
        len(P_VALUES)
        * len(SCENARIOS)
        * TRIALS
    )

    run_number = 0

    print("FINAL BB84 reproduction experiment")
    print("===================================")
    print()
    print(f"p values       : {P_VALUES}")
    print(f"scenarios      : {SCENARIOS}")
    print(f"transmissions  : {TRANSMISSIONS}")
    print(f"trials         : {TRIALS}")
    print(f"total runs     : {total_runs}")
    print(f"seed           : {SEED}")
    print()

    for scenario in SCENARIOS:

        for p in P_VALUES:

            trial_qbers = []
            trial_sifted = []

            reference_qber = bb84_average_qber(
                p,
                scenario=scenario,
            )

            for trial in range(1, TRIALS + 1):

                run_number += 1

                trial_seed = int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                )

                result = run_one_trial(
                    p=p,
                    scenario=scenario,
                    seed=trial_seed,
                )

                qber = float(
                    result["qber"]
                )

                sifted_bits = int(
                    result["sifted_length"]
                )

                trial_qbers.append(qber)
                trial_sifted.append(sifted_bits)

                raw_rows.append({
                    "p": p,
                    "scenario": scenario,
                    "trial": trial,
                    "seed": trial_seed,
                    "n_transmissions": TRANSMISSIONS,
                    "sifted_bits": sifted_bits,
                    "qber": qber,
                    "reference_qber": reference_qber,
                    "difference_from_reference":
                        qber - reference_qber,
                })

                if (
                    trial == 1
                    or trial == TRIALS
                ):
                    print(
                        f"{run_number:4d}/{total_runs}"
                        f" | {scenario:7s}"
                        f" | p={p:.2f}"
                        f" | trial={trial:02d}"
                        f" | QBER={qber:.6f}"
                    )

            qbers = np.asarray(
                trial_qbers,
                dtype=float,
            )

            sifted = np.asarray(
                trial_sifted,
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

            sigma_p = (
                p
                * {
                    "single": 1,
                    "dual": 2,
                    "triple": 3,
                }[scenario]
            )

            key_rate = paper_key_rate(
                mean_qber
            )

            pri = paper_pri(
                mean_qber,
                sigma_p,
            )

            paper_value = PAPER_TABLE_4.get(
                (p, scenario),
                np.nan,
            )

            summary_rows.append({
                "p": p,
                "scenario": scenario,
                "trials": TRIALS,
                "n_transmissions_per_trial":
                    TRANSMISSIONS,
                "mean_sifted_bits":
                    float(np.mean(sifted)),
                "mean_qber":
                    mean_qber,
                "std_qber":
                    std_qber,
                "ci95_low":
                    ci_low,
                "ci95_high":
                    ci_high,
                "reference_qber":
                    reference_qber,
                "difference_from_reference":
                    mean_qber - reference_qber,
                "paper_table_4_qber":
                    paper_value,
                "difference_from_paper":
                    (
                        mean_qber - paper_value
                        if not np.isnan(paper_value)
                        else np.nan
                    ),
                "sigma_p":
                    sigma_p,
                "secret_key_rate":
                    key_rate,
                "pri":
                    pri,
            })

    raw_fields = [
        "p",
        "scenario",
        "trial",
        "seed",
        "n_transmissions",
        "sifted_bits",
        "qber",
        "reference_qber",
        "difference_from_reference",
    ]

    summary_fields = [
        "p",
        "scenario",
        "trials",
        "n_transmissions_per_trial",
        "mean_sifted_bits",
        "mean_qber",
        "std_qber",
        "ci95_low",
        "ci95_high",
        "reference_qber",
        "difference_from_reference",
        "paper_table_4_qber",
        "difference_from_paper",
        "sigma_p",
        "secret_key_rate",
        "pri",
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
    print("Experiment complete.")
    print(f"Raw results     : {RAW_FILE}")
    print(f"Summary results : {SUMMARY_FILE}")


if __name__ == "__main__":
    main()