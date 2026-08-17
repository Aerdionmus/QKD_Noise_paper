import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_bb84"
    / "bb84_final_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_bb84"
    / "figures"
)


SCENARIOS = [
    "single",
    "dual",
    "triple",
]


SCENARIO_LABELS = {
    "single": "Single",
    "dual": "Dual",
    "triple": "Triple",
}


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


def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for_scenario(rows, scenario):
    selected = [
        row
        for row in rows
        if row["scenario"] == scenario
    ]

    selected.sort(
        key=lambda row: float(row["p"])
    )

    return selected


def plot_qber_vs_p(rows):
    plt.figure(figsize=(8, 5.5))

    for scenario in SCENARIOS:
        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array(
            [float(row["p"]) for row in data]
        )

        qber = np.array(
            [float(row["mean_qber"]) for row in data]
        )

        ci_low = np.array(
            [float(row["ci95_low"]) for row in data]
        )

        ci_high = np.array(
            [float(row["ci95_high"]) for row in data]
        )

        plt.plot(
            p,
            qber,
            marker="o",
            label=SCENARIO_LABELS[scenario],
        )

        plt.fill_between(
            p,
            ci_low,
            ci_high,
            alpha=0.15,
        )

    plt.xlabel("Noise parameter p")
    plt.ylabel("QBER")
    plt.title("BB84 QBER versus composite-channel noise")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "qber_vs_p.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def plot_analytical_vs_aer(rows):
    plt.figure(figsize=(7, 7))

    all_reference = []
    all_measured = []

    for scenario in SCENARIOS:
        data = rows_for_scenario(
            rows,
            scenario,
        )

        reference = np.array(
            [
                float(row["reference_qber"])
                for row in data
            ]
        )

        measured = np.array(
            [
                float(row["mean_qber"])
                for row in data
            ]
        )

        all_reference.extend(reference)
        all_measured.extend(measured)

        plt.scatter(
            reference,
            measured,
            label=SCENARIO_LABELS[scenario],
            s=45,
        )

    all_reference = np.array(all_reference)
    all_measured = np.array(all_measured)

    lower = min(
        all_reference.min(),
        all_measured.min(),
    )

    upper = max(
        all_reference.max(),
        all_measured.max(),
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect agreement",
    )

    plt.xlabel("Analytical QBER")
    plt.ylabel("Aer mean QBER")
    plt.title("Analytical model versus Qiskit Aer")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "analytical_vs_aer.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def plot_paper_comparison(rows):
    plt.figure(figsize=(8, 5.5))

    for scenario in ["single", "dual"]:
        data = [
            row
            for row in rows
            if (
                row["scenario"] == scenario
                and row["paper_table_4_qber"].lower()
                != "nan"
            )
        ]

        data.sort(
            key=lambda row: float(row["p"])
        )

        p = np.array(
            [float(row["p"]) for row in data]
        )

        ours = np.array(
            [float(row["mean_qber"]) for row in data]
        )

        paper = np.array(
            [
                float(row["paper_table_4_qber"])
                for row in data
            ]
        )

        plt.plot(
            p,
            ours,
            marker="o",
            label=f"Our {SCENARIO_LABELS[scenario]}",
        )

        plt.plot(
            p,
            paper,
            marker="x",
            linestyle="--",
            label=f"Paper {SCENARIO_LABELS[scenario]}",
        )

    plt.xlabel("Noise parameter p")
    plt.ylabel("QBER")
    plt.title("Track-A reproduction versus published Table 4")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "paper_table4_comparison.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def plot_secret_key_rate(rows):
    plt.figure(figsize=(8, 5.5))

    for scenario in SCENARIOS:
        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array(
            [float(row["p"]) for row in data]
        )

        key_rate = np.array(
            [
                float(row["secret_key_rate"])
                for row in data
            ]
        )

        plt.plot(
            p,
            key_rate,
            marker="o",
            label=SCENARIO_LABELS[scenario],
        )

    plt.xlabel("Noise parameter p")
    plt.ylabel("Secret-key rate R")
    plt.title("Paper secret-key-rate expression")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "secret_key_rate_vs_p.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def plot_pri(rows):
    plt.figure(figsize=(8, 5.5))

    for scenario in SCENARIOS:
        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array(
            [float(row["p"]) for row in data]
        )

        pri = np.array(
            [float(row["pri"]) for row in data]
        )

        plt.plot(
            p,
            pri,
            marker="o",
            label=SCENARIO_LABELS[scenario],
        )

    plt.xlabel("Noise parameter p")
    plt.ylabel("PRI")
    plt.title("Paper PRI expression")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "pri_vs_p.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def main():
    print("Final BB84 figure generation")
    print("============================")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_results()

    if len(rows) != 30:
        raise RuntimeError(
            f"Expected 30 summary rows, found {len(rows)}"
        )

    print(f"Input rows : {len(rows)}")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    plot_qber_vs_p(rows)
    plot_analytical_vs_aer(rows)
    plot_paper_comparison(rows)
    plot_secret_key_rate(rows)
    plot_pri(rows)

    print()
    print("Figure generation complete.")


if __name__ == "__main__":
    main()
