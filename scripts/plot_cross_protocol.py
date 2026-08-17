import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "cross_protocol"
    / "cross_protocol_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cross_protocol"
    / "figures"
)


PROTOCOLS = ["BB84", "B92", "E91"]
SCENARIOS = ["single", "dual", "triple"]


def load_results():
    """Load cross-protocol summary CSV."""

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def rows_for_scenario(rows, scenario):
    """Return rows for one noise scenario, ordered by p."""

    selected = [
        row
        for row in rows
        if row["scenario"] == scenario
    ]

    selected.sort(
        key=lambda row: float(row["p"])
    )

    return selected


def save_figure(filename):
    """Save current figure and close it."""

    output = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output}")


def plot_qber_comparison(rows):
    """Plot QBER for all three protocols."""

    plt.figure(figsize=(10, 6))

    for scenario in SCENARIOS:
        scenario_rows = rows_for_scenario(
            rows,
            scenario,
        )

        p_values = [
            float(row["p"])
            for row in scenario_rows
        ]

        for protocol in PROTOCOLS:
            qber_values = [
                float(row[f"{protocol.lower()}_qber"])
                for row in scenario_rows
            ]

            plt.plot(
                p_values,
                qber_values,
                marker="o",
                label=f"{protocol} ({scenario})",
            )

    plt.xlabel("Noise probability p")
    plt.ylabel("QBER")
    plt.title("QBER Comparison Across QKD Protocols")
    plt.grid(True, alpha=0.3)
    plt.legend(
        ncol=2,
        fontsize=8,
    )

    save_figure("qber_comparison.png")


def plot_bb84_key_rate(rows):
    """Plot BB84 secret-key rate."""

    plt.figure(figsize=(10, 6))

    for scenario in SCENARIOS:
        scenario_rows = rows_for_scenario(
            rows,
            scenario,
        )

        p_values = [
            float(row["p"])
            for row in scenario_rows
        ]

        key_rates = [
            float(row["bb84_secret_key_rate"])
            for row in scenario_rows
        ]

        plt.plot(
            p_values,
            key_rates,
            marker="o",
            label=scenario.capitalize(),
        )

    plt.xlabel("Noise probability p")
    plt.ylabel("BB84 secret-key rate")
    plt.title("BB84 Secret-Key Rate Under Composite Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("bb84_key_rate.png")


def plot_b92_conclusive_fraction(rows):
    """Plot B92 conclusive fraction."""

    # The cross-protocol CSV currently does not contain
    # B92 conclusive fraction, so this figure is intentionally
    # generated from the original B92 final summary.

    input_file = (
        PROJECT_ROOT
        / "results"
        / "final_b92"
        / "b92_final_summary.csv"
    )

    with input_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        b92_rows = list(csv.DictReader(handle))

    plt.figure(figsize=(10, 6))

    for scenario in SCENARIOS:
        scenario_rows = [
            row
            for row in b92_rows
            if row["scenario"] == scenario
        ]

        scenario_rows.sort(
            key=lambda row: float(row["p"])
        )

        p_values = [
            float(row["p"])
            for row in scenario_rows
        ]

        conclusive_values = [
            float(row["mean_conclusive_fraction"])
            for row in scenario_rows
        ]

        plt.plot(
            p_values,
            conclusive_values,
            marker="o",
            label=scenario.capitalize(),
        )

    plt.xlabel("Noise probability p")
    plt.ylabel("Mean conclusive fraction")
    plt.title("B92 Conclusive Fraction Under Composite Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("b92_conclusive_fraction.png")


def plot_e91_chsh(rows):
    """Plot E91 CHSH Bell parameter."""

    plt.figure(figsize=(10, 6))

    for scenario in SCENARIOS:
        scenario_rows = rows_for_scenario(
            rows,
            scenario,
        )

        p_values = [
            float(row["p"])
            for row in scenario_rows
        ]

        chsh_values = [
            float(row["e91_chsh"])
            for row in scenario_rows
        ]

        plt.plot(
            p_values,
            chsh_values,
            marker="o",
            label=scenario.capitalize(),
        )

    plt.axhline(
        2.0,
        linestyle="--",
        label="CHSH classical bound",
    )

    plt.xlabel("Noise probability p")
    plt.ylabel("CHSH S")
    plt.title("E91 CHSH Violation Under Composite Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("e91_chsh.png")


def main():
    print("Cross-protocol figure generation")
    print("================================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_results()

    print(f"Input rows : {len(rows)}")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    if len(rows) != 30:
        raise ValueError(
            f"Expected 30 comparison rows, "
            f"found {len(rows)}."
        )

    plot_qber_comparison(rows)
    plot_bb84_key_rate(rows)
    plot_b92_conclusive_fraction(rows)
    plot_e91_chsh(rows)

    print()
    print("Figure generation complete.")


if __name__ == "__main__":
    main()