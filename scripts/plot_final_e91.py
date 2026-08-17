import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_e91"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "e91_final_summary.csv"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


SCENARIOS = [
    "single",
    "dual",
    "triple",
]


def load_results():
    with SUMMARY_FILE.open(
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def rows_for_scenario(
    rows,
    scenario,
):
    selected = [
        row
        for row in rows
        if row["scenario"] == scenario
    ]

    selected.sort(
        key=lambda row: float(
            row["p"]
        )
    )

    return selected


def plot_qber_vs_p(rows):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for scenario in SCENARIOS:

        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array([
            float(row["p"])
            for row in data
        ])

        mean_qber = np.array([
            float(row["mean_qber"])
            for row in data
        ])

        exact_qber = np.array([
            float(row["mean_qber"])
            for row in data
        ])

        ci_low = np.array([
            float(row["ci95_qber_low"])
            for row in data
        ])

        ci_high = np.array([
            float(row["ci95_qber_high"])
            for row in data
        ])

        ax.plot(
            p,
            mean_qber,
            marker="o",
            label=scenario,
        )

        ax.fill_between(
            p,
            ci_low,
            ci_high,
            alpha=0.15,
        )

    ax.axhline(
        0.146,
        linestyle="--",
        label="QBER threshold = 0.146",
    )

    ax.set_xlabel(
        "Noise parameter p"
    )

    ax.set_ylabel(
        "QBER"
    )

    ax.set_title(
        "E91 QBER vs Noise Parameter"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "qber_vs_p.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Created: {output}"
    )


def plot_chsh_vs_p(rows):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for scenario in SCENARIOS:

        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array([
            float(row["p"])
            for row in data
        ])

        chsh = np.array([
            float(row["exact_chsh"])
            for row in data
        ])

        ax.plot(
            p,
            chsh,
            marker="o",
            label=scenario,
        )

    ax.axhline(
        2.0,
        linestyle="--",
        label="Bell threshold S = 2",
    )

    ax.axhline(
        2.828427124746,
        linestyle=":",
        label="Tsirelson bound",
    )

    ax.set_xlabel(
        "Noise parameter p"
    )

    ax.set_ylabel(
        "CHSH S"
    )

    ax.set_title(
        "E91 CHSH Bell Parameter vs Noise"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "chsh_vs_p.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Created: {output}"
    )


def plot_qber_confidence_intervals(
    rows,
):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for scenario in SCENARIOS:

        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array([
            float(row["p"])
            for row in data
        ])

        mean = np.array([
            float(row["mean_qber"])
            for row in data
        ])

        low = np.array([
            float(row["ci95_qber_low"])
            for row in data
        ])

        high = np.array([
            float(row["ci95_qber_high"])
            for row in data
        ])

        ax.errorbar(
            p,
            mean,
            yerr=[
                mean - low,
                high - mean,
            ],
            marker="o",
            capsize=4,
            label=scenario,
        )

    ax.axhline(
        0.146,
        linestyle="--",
        label="QBER threshold = 0.146",
    )

    ax.set_xlabel(
        "Noise parameter p"
    )

    ax.set_ylabel(
        "Mean QBER"
    )

    ax.set_title(
        "E91 QBER with 95% Confidence Intervals"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "qber_confidence_intervals.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Created: {output}"
    )


def plot_secure_fraction(rows):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for scenario in SCENARIOS:

        data = rows_for_scenario(
            rows,
            scenario,
        )

        p = np.array([
            float(row["p"])
            for row in data
        ])

        secure_fraction = np.array([
            float(
                row["secure_fraction"]
            )
            for row in data
        ])

        ax.plot(
            p,
            secure_fraction,
            marker="o",
            label=scenario,
        )

    ax.set_xlabel(
        "Noise parameter p"
    )

    ax.set_ylabel(
        "Secure-trial fraction"
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_title(
        "E91 Secure-Trial Fraction vs Noise"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / "secure_fraction_vs_p.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Created: {output}"
    )


def main():

    print(
        "Final E91 figure generation"
    )

    print(
        "============================"
    )

    print()

    rows = load_results()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Input rows : {len(rows)}"
    )

    print(
        f"Output dir : {FIGURES_DIR}"
    )

    print()

    plot_qber_vs_p(rows)

    plot_chsh_vs_p(rows)

    plot_qber_confidence_intervals(
        rows
    )

    plot_secure_fraction(rows)

    print()

    print(
        "Figure generation complete."
    )


if __name__ == "__main__":
    main()