import csv
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_bb84"
    / "bb84_final_summary.csv"
)


def load_summary():
    with SUMMARY_FILE.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_summary()

    print("FINAL BB84 STATISTICAL AUDIT")
    print("============================")
    print()
    print(f"Summary rows : {len(rows)}")
    print()

    if len(rows) != 30:
        raise RuntimeError(
            f"Expected 30 summary rows, found {len(rows)}"
        )

    differences = []
    outside_ci = []

    print(
        "scenario   p       reference"
        "       mean        difference"
        "       95% CI"
    )
    print("-" * 85)

    for row in rows:
        scenario = row["scenario"]
        p = float(row["p"])

        reference = float(row["reference_qber"])
        mean = float(row["mean_qber"])
        difference = float(row["difference_from_reference"])

        ci_low = float(row["ci95_low"])
        ci_high = float(row["ci95_high"])

        differences.append(abs(difference))

        inside = (
            ci_low <= reference <= ci_high
        )

        if not inside:
            outside_ci.append(
                (
                    scenario,
                    p,
                    reference,
                    mean,
                    ci_low,
                    ci_high,
                )
            )

        print(
            f"{scenario:8s}"
            f" {p:0.2f}"
            f"    {reference:0.6f}"
            f"    {mean:0.6f}"
            f"    {difference:+0.6f}"
            f"    [{ci_low:0.6f}, {ci_high:0.6f}]"
        )

    print()
    print("SUMMARY")
    print("=======")

    max_difference = max(differences)
    mean_absolute_difference = float(
        np.mean(differences)
    )

    print(
        f"Maximum absolute Aer-reference difference"
        f" : {max_difference:.6f}"
    )

    print(
        f"Mean absolute Aer-reference difference"
        f"    : {mean_absolute_difference:.6f}"
    )

    print(
        f"95% CIs containing analytical reference"
        f" : {30 - len(outside_ci)}/30"
    )

    print()

    if outside_ci:
        print("REFERENCE VALUES OUTSIDE 95% CI")
        print("===============================")

        for (
            scenario,
            p,
            reference,
            mean,
            ci_low,
            ci_high,
        ) in outside_ci:

            print(
                f"{scenario:8s}"
                f" p={p:.2f}"
                f" reference={reference:.6f}"
                f" mean={mean:.6f}"
                f" CI=[{ci_low:.6f}, {ci_high:.6f}]"
            )

    else:
        print(
            "All 30 analytical reference values are"
            " inside their corresponding 95% CIs."
        )

    print()
    print("PAPER TABLE 4 DISCREPANCIES")
    print("============================")

    paper_rows = [
        row
        for row in rows
        if row["paper_table_4_qber"].lower() != "nan"
    ]

    paper_differences = []

    for row in paper_rows:
        scenario = row["scenario"]
        p = float(row["p"])

        ours = float(row["mean_qber"])
        paper = float(row["paper_table_4_qber"])

        difference = ours - paper

        paper_differences.append(
            abs(difference)
        )

        print(
            f"{scenario:8s}"
            f" p={p:.2f}"
            f" ours={ours:.6f}"
            f" paper={paper:.6f}"
            f" difference={difference:+.6f}"
        )

    print()

    if paper_differences:
        print(
            f"Mean absolute difference from paper Table 4"
            f" : {np.mean(paper_differences):.6f}"
        )

        print(
            f"Maximum absolute difference from paper Table 4"
            f" : {max(paper_differences):.6f}"
        )

    print()
    print("AUDIT COMPLETE.")


if __name__ == "__main__":
    main()
