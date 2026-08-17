import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUTS = {
    "BB84": PROJECT_ROOT / "results" / "final_bb84" / "bb84_final_summary.csv",
    "B92": PROJECT_ROOT / "results" / "final_b92" / "b92_final_summary.csv",
    "E91": PROJECT_ROOT / "results" / "final_e91" / "e91_final_summary.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_protocol"
OUTPUT_FILE = OUTPUT_DIR / "cross_protocol_summary.csv"
AUDIT_FILE = OUTPUT_DIR / "cross_protocol_audit.txt"


SCENARIOS = ["single", "dual", "triple"]
P_VALUES = [round(x, 2) for x in np.arange(0.01, 0.101, 0.01)]


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_value(row, *names, default=np.nan):
    for name in names:
        if name in row and row[name] not in ("", "nan", "NaN"):
            try:
                return float(row[name])
            except ValueError:
                pass
    return default


def load_all():
    data = {}

    for protocol, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {protocol} file: {path}")

        rows = load_csv(path)

        if len(rows) != 30:
            raise RuntimeError(
                f"{protocol}: expected 30 rows, found {len(rows)}"
            )

        data[protocol] = rows

    return data


def index_rows(rows):
    result = {}

    for row in rows:
        p = round(float(row["p"]), 2)
        scenario = row["scenario"]
        result[(p, scenario)] = row

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    indexed = {
        protocol: index_rows(rows)
        for protocol, rows in data.items()
    }

    output_rows = []

    for p in P_VALUES:
        for scenario in SCENARIOS:
            key = (p, scenario)

            if any(key not in indexed[protocol] for protocol in indexed):
                raise RuntimeError(
                    f"Missing comparison row for p={p}, scenario={scenario}"
                )

            bb84 = indexed["BB84"][key]
            b92 = indexed["B92"][key]
            e91 = indexed["E91"][key]

            bb84_qber = find_value(bb84, "mean_qber")
            b92_qber = find_value(b92, "mean_qber")
            e91_qber = find_value(e91, "mean_qber")

            bb84_secure = find_value(
                bb84,
                "secure_fraction",
                "security_fraction",
                default=np.nan,
            )

            b92_secure = find_value(
                b92,
                "secure_fraction",
                "security_fraction",
                default=np.nan,
            )

            e91_secure = find_value(
                e91,
                "secure_fraction",
                "security_fraction",
                default=np.nan,
            )

            bb84_rate = find_value(
                bb84,
                "secret_key_rate",
                "key_rate",
                default=np.nan,
            )

            b92_rate = find_value(
                b92,
                "secret_key_rate",
                "key_rate",
                default=np.nan,
            )

            e91_chsh = find_value(
                e91,
                "mean_chsh",
                "chsh_s",
                "exact_chsh",
                default=np.nan,
            )

            output_rows.append(
                {
                    "p": p,
                    "scenario": scenario,

                    "bb84_qber": bb84_qber,
                    "bb84_qber_ci_low": find_value(
                        bb84, "ci95_low"
                    ),
                    "bb84_qber_ci_high": find_value(
                        bb84, "ci95_high"
                    ),
                    "bb84_secure_fraction": bb84_secure,
                    "bb84_secret_key_rate": bb84_rate,

                    "b92_qber": b92_qber,
                    "b92_qber_ci_low": find_value(
                        b92,
                        "ci95_low",
                        "bootstrap_ci95_low",
                    ),
                    "b92_qber_ci_high": find_value(
                        b92,
                        "ci95_high",
                        "bootstrap_ci95_high",
                    ),
                    "b92_secure_fraction": b92_secure,
                    "b92_secret_key_rate": b92_rate,

                    "e91_qber": e91_qber,
                    "e91_qber_ci_low": find_value(
                        e91, "ci95_qber_low", "ci95_low"
                    ),
                    "e91_qber_ci_high": find_value(
                        e91, "ci95_qber_high", "ci95_high"
                    ),
                    "e91_secure_fraction": e91_secure,
                    "e91_chsh": e91_chsh,
                }
            )

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # --------------------------------------------------------------
    # Audit
    # --------------------------------------------------------------

    audit_lines = []

    audit_lines.append("CROSS-PROTOCOL ANALYSIS AUDIT")
    audit_lines.append("=" * 70)
    audit_lines.append("")

    for protocol, rows in data.items():
        audit_lines.append(
            f"{protocol}: {len(rows)} summary rows"
        )

    audit_lines.append("")
    audit_lines.append(
        f"Comparison rows generated: {len(output_rows)}"
    )

    # QBER ordering
    audit_lines.append("")
    audit_lines.append("QBER COMPARISON")
    audit_lines.append("-" * 70)

    for scenario in SCENARIOS:
        audit_lines.append("")
        audit_lines.append(f"Scenario: {scenario}")

        for p in P_VALUES:
            row = next(
                r for r in output_rows
                if r["p"] == p and r["scenario"] == scenario
            )

            values = {
                "BB84": row["bb84_qber"],
                "B92": row["b92_qber"],
                "E91": row["e91_qber"],
            }

            ordered = sorted(
                values.items(),
                key=lambda item: item[1]
            )

            ranking = " < ".join(
                f"{name} ({value:.6f})"
                for name, value in ordered
            )

            audit_lines.append(
                f"p={p:.2f}: {ranking}"
            )

    # Best/worst protocol at each point
    audit_lines.append("")
    audit_lines.append("BEST/WORST QBER")
    audit_lines.append("-" * 70)

    for scenario in SCENARIOS:
        audit_lines.append("")
        audit_lines.append(f"Scenario: {scenario}")

        for p in P_VALUES:
            row = next(
                r for r in output_rows
                if r["p"] == p and r["scenario"] == scenario
            )

            values = {
                "BB84": row["bb84_qber"],
                "B92": row["b92_qber"],
                "E91": row["e91_qber"],
            }

            best = min(values, key=values.get)
            worst = max(values, key=values.get)

            audit_lines.append(
                f"p={p:.2f}: "
                f"best={best} ({values[best]:.6f}), "
                f"worst={worst} ({values[worst]:.6f})"
            )

    # E91 CHSH
    audit_lines.append("")
    audit_lines.append("E91 CHSH")
    audit_lines.append("-" * 70)

    for scenario in SCENARIOS:
        values = []

        for p in P_VALUES:
            row = next(
                r for r in output_rows
                if r["p"] == p and r["scenario"] == scenario
            )

            values.append(
                (p, row["e91_chsh"])
            )

        audit_lines.append(f"{scenario}:")
        for p, value in values:
            audit_lines.append(
                f"  p={p:.2f}: S={value:.6f}"
            )

    # BB84 key rate
    audit_lines.append("")
    audit_lines.append("BB84 SECRET-KEY RATE")
    audit_lines.append("-" * 70)

    for scenario in SCENARIOS:
        audit_lines.append(f"{scenario}:")

        for p in P_VALUES:
            row = next(
                r for r in output_rows
                if r["p"] == p and r["scenario"] == scenario
            )

            value = row["bb84_secret_key_rate"]

            audit_lines.append(
                f"  p={p:.2f}: "
                f"R={value:.6f}"
            )

    audit_lines.append("")
    audit_lines.append("AUDIT COMPLETE.")

    AUDIT_FILE.write_text(
        "\n".join(audit_lines) + "\n"
    )

    print("Cross-protocol analysis")
    print("=======================")
    print()
    print("Protocols : BB84, B92, E91")
    print("Rows/protocol : 30")
    print("Comparison rows : 30")
    print()
    print(f"Created: {OUTPUT_FILE}")
    print(f"Created: {AUDIT_FILE}")
    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
