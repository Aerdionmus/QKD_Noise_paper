from qkd_noise.protocols.b92 import simulate_b92


def main():
    print("B92 implementation smoke test")
    print("=============================")

    scenarios = ["single", "dual", "triple"]

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")

        result = simulate_b92(
            n=1000,
            p=0.05,
            scenario=scenario,
            seed=12345,
        )

        print(
            f"QBER={result['qber']:.6f}"
            f" | conclusive={result['conclusive_count']}"
            f" | fraction={result['conclusive_fraction']:.6f}"
            f" | total={result['total_bits']}"
        )

    print("\nB92 smoke test complete.")


if __name__ == "__main__":
    main()