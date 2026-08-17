import csv
from pathlib import Path


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
    / "tables"
)

OUTPUT_CSV = OUTPUT_DIR / "paper_summary_table.csv"
OUTPUT_MD = OUTPUT_DIR / "paper_summary_table.md"


REPRESENTATIVE_P_VALUES = [0.01, 0.05, 0.10]
SCENARIOS = ["single", "dual", "triple"]


def load_rows():
    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def find_row(rows, p, scenario):
    for row in rows:
        if (
            abs(float(row["p"]) - p) < 1e-12
            and row["scenario"] == scenario
        ):
            return row

    raise ValueError(
        f"Missing row for p={p}, scenario={scenario}"
    )


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def build_table(rows):
    output = []

    for p in REPRESENTATIVE_P_VALUES:
        for scenario in SCENARIOS:
            row = find_row(
                rows,
                p,
                scenario,
            )

            output.append(
                {
                    "p": p,
                    "scenario": scenario,

                    "BB84 QBER": fmt(
                        row["bb84_qber"]
                    ),

                    "BB84 CI95": (
                        f"[{fmt(row['bb84_qber_ci_low'])}, "
                        f"{fmt(row['bb84_qber_ci_high'])}]"
                    ),

                    "BB84 key rate": fmt(
                        row["bb84_secret_key_rate"]
                    ),

                    "B92 QBER": fmt(
                        row["b92_qber"]
                    ),

                    "B92 CI95": (
                        f"[{fmt(row['b92_qber_ci_low'])}, "
                        f"{fmt(row['b92_qber_ci_high'])}]"
                    ),

                    "E91 QBER": fmt(
                        row["e91_qber"]
                    ),

                    "E91 CI95": (
                        f"[{fmt(row['e91_qber_ci_low'])}, "
                        f"{fmt(row['e91_qber_ci_high'])}]"
                    ),

                    "E91 CHSH": fmt(
                        row["e91_chsh"]
                    ),
                }
            )

    return output


def write_csv(table):
    fields = [
        "p",
        "scenario",
        "BB84 QBER",
        "BB84 CI95",
        "BB84 key rate",
        "B92 QBER",
        "B92 CI95",
        "E91 QBER",
        "E91 CI95",
        "E91 CHSH",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(table)


def write_markdown(table):
    headers = [
        "p",
        "Scenario",
        "BB84 QBER",
        "BB84 95% CI",
        "BB84 key rate",
        "B92 QBER",
        "B92 95% CI",
        "E91 QBER",
        "E91 95% CI",
        "E91 CHSH",
    ]

    lines = []

    lines.append(
        "| "
        + " | ".join(headers)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(["---"] * len(headers))
        + " |"
    )

    for row in table:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['p']:.2f}",
                    row["scenario"],
                    row["BB84 QBER"],
                    row["BB84 CI95"],
                    row["BB84 key rate"],
                    row["B92 QBER"],
                    row["B92 CI95"],
                    row["E91 QBER"],
                    row["E91 CI95"],
                    row["E91 CHSH"],
                ]
            )
            + " |"
        )

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    print("Paper-quality summary table generation")
    print("========================================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    print(f"Input rows : {len(rows)}")

    if len(rows) != 30:
        raise ValueError(
            f"Expected 30 rows, found {len(rows)}."
        )

    table = build_table(rows)

    write_csv(table)
    write_markdown(table)

    print(f"Created: {OUTPUT_CSV}")
    print(f"Created: {OUTPUT_MD}")
    print()
    print(
        f"Representative p values : "
        f"{REPRESENTATIVE_P_VALUES}"
    )
    print(
        f"Scenarios                : "
        f"{SCENARIOS}"
    )
    print(
        f"Table rows               : "
        f"{len(table)}"
    )
    print()
    print("Table generation complete.")


if __name__ == "__main__":
    main()