import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "final_b92"
SUMMARY_FILE = RESULTS_DIR / "b92_final_summary.csv"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results():
    with SUMMARY_FILE.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for_scenario(rows, scenario):
    selected = [
        r for r in rows
        if r["scenario"] == scenario
    ]

    selected.sort(
        key=lambda r: float(r["p"])
    )

    return selected


def plot_qber_vs_p(rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    for scenario in ["single", "dual", "triple"]:
        data = rows_for_scenario(rows, scenario)

        p = np.array(
            [float(r["p"]) for r in data]
        )

        mean = np.array(
            [float(r["mean_qber"]) for r in data]
        )

        exact = np.array(
            [float(r["exact_qber"]) for r in data]
        )

        ax.plot(
            p,
            mean,
            marker="o",
            label=f"{scenario} Monte Carlo",
        )

        ax.plot(
            p,
            exact,
            linestyle="--",
            label=f"{scenario} analytical",
        )

    ax.set_xlabel("Noise parameter p")
    ax.set_ylabel("B92 QBER")
    ax.set_title("B92 QBER vs Noise Parameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "qber_vs_p.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Created: {output}")


def plot_analytical_vs_mc(rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    for scenario in ["single", "dual", "triple"]:
        data = rows_for_scenario(rows, scenario)

        exact = np.array(
            [float(r["exact_qber"]) for r in data]
        )

        mean = np.array(
            [float(r["mean_qber"]) for r in data]
        )

        ax.plot(
            exact,
            mean,
            marker="o",
            label=scenario,
        )

    all_exact = np.array(
        [float(r["exact_qber"]) for r in rows]
    )

    low = float(np.min(all_exact))
    high = float(np.max(all_exact))

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        label="y = x",
    )

    ax.set_xlabel("Analytical B92 QBER")
    ax.set_ylabel("Monte Carlo mean QBER")
    ax.set_title("B92 Analytical vs Monte Carlo QBER")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "analytical_vs_mc.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Created: {output}")


def plot_conclusive_fraction(rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    for scenario in ["single", "dual", "triple"]:
        data = rows_for_scenario(rows, scenario)

        p = np.array(
            [float(r["p"]) for r in data]
        )

        mean = np.array(
            [float(r["mean_conclusive_fraction"]) for r in data]
        )

        exact = np.array(
            [float(r["exact_conclusive_probability"]) for r in data]
        )

        ax.plot(
            p,
            mean,
            marker="o",
            label=f"{scenario} Monte Carlo",
        )

        ax.plot(
            p,
            exact,
            linestyle="--",
            label=f"{scenario} analytical",
        )

    ax.set_xlabel("Noise parameter p")
    ax.set_ylabel("Conclusive fraction")
    ax.set_title("B92 Conclusive Fraction vs Noise Parameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "conclusive_fraction_vs_p.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Created: {output}")


def plot_qber_difference(rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    for scenario in ["single", "dual", "triple"]:
        data = rows_for_scenario(rows, scenario)

        p = np.array(
            [float(r["p"]) for r in data]
        )

        difference = np.array(
            [float(r["difference_from_exact"]) for r in data]
        )

        ax.plot(
            p,
            difference,
            marker="o",
            label=scenario,
        )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_xlabel("Noise parameter p")
    ax.set_ylabel("Monte Carlo − analytical QBER")
    ax.set_title("B92 QBER Difference from Analytical Reference")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "qber_difference_vs_p.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Created: {output}")


def main():
    print("Final B92 figure generation")
    print("============================")
    print()

    rows = load_results()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input rows : {len(rows)}")
    print(f"Output dir : {FIGURES_DIR}")
    print()

    plot_qber_vs_p(rows)
    plot_analytical_vs_mc(rows)
    plot_conclusive_fraction(rows)
    plot_qber_difference(rows)

    print()
    print("Figure generation complete.")


if __name__ == "__main__":
    main()