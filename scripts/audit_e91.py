import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.protocols.e91 import (
    ALICE_BASES,
    BOB_BASES,
    CHSH_SETTINGS,
    E91_VISIBILITY,
    apply_noise,
    chsh_value,
    correlation,
    phi_plus,
    projectors,
    sample_measurement,
    simulate_e91,
    werner_state,
)


def print_correlations(rho):
    print()
    print("E91 correlation matrix")
    print("=" * 70)
    print("Alice/Bob basis angles")
    print()

    print("             ", end="")
    for bob in BOB_BASES:
        print(f"{bob:>12.1f}°", end="")
    print()

    for alice in ALICE_BASES:
        print(f"Alice {alice:>5.1f}°", end="")

        for bob in BOB_BASES:
            value = correlation(
                rho,
                alice,
                bob,
            )

            print(
                f"{value:>13.6f}",
                end="",
            )

        print()
def monte_carlo_chsh(
    rho,
    settings,
    samples_per_setting=100000,
    seed=20260817,
):
    """Fast independent Monte-Carlo estimate of CHSH S."""

    rng = np.random.default_rng(seed)

    pairs = {
        "ab": (settings["a"], settings["b"]),
        "ab_prime": (settings["a"], settings["b_prime"]),
        "a_prime_b": (settings["a_prime"], settings["b"]),
        "a_prime_b_prime": (
            settings["a_prime"],
            settings["b_prime"],
        ),
    }

    correlations = {}

    for name, (alice_angle, bob_angle) in pairs.items():

        alice_plus, alice_minus = projectors(alice_angle)
        bob_plus, bob_minus = projectors(bob_angle)

        projectors_a = {
            0: alice_plus,
            1: alice_minus,
        }

        projectors_b = {
            0: bob_plus,
            1: bob_minus,
        }

        probabilities = np.zeros((2, 2), dtype=float)

        for alice_bit in (0, 1):
            for bob_bit in (0, 1):

                projector = np.kron(
                    projectors_a[alice_bit],
                    projectors_b[bob_bit],
                )

                probability = np.trace(
                    projector @ rho
                )

                probabilities[alice_bit, bob_bit] = max(
                    0.0,
                    float(np.real_if_close(probability)),
                )

        probabilities /= probabilities.sum()

        outcomes = rng.choice(
            4,
            size=samples_per_setting,
            p=probabilities.reshape(-1),
        )

        alice_bits = outcomes // 2
        bob_bits = outcomes % 2

        alice_values = np.where(
            alice_bits == 0,
            1,
            -1,
        )

        bob_values = np.where(
            bob_bits == 0,
            1,
            -1,
        )

        correlations[name] = float(
            np.mean(
                alice_values * bob_values
            )
        )

    s_mc = abs(
        correlations["ab"]
        + correlations["ab_prime"]
        + correlations["a_prime_b"]
        - correlations["a_prime_b_prime"]
    )

    return s_mc, correlations
def main():
    print("E91 mathematical / protocol audit")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Ideal Bell state
    # ------------------------------------------------------------------

    ideal = phi_plus()

    print("1. Ideal Bell state")
    print("-" * 70)

    trace = np.trace(ideal)

    hermitian_error = np.max(
        np.abs(
            ideal - ideal.conj().T
        )
    )

    eigenvalues = np.linalg.eigvalsh(ideal)

    print(
        f"Trace              : {trace.real:.12f}"
    )

    print(
        f"Hermitian error    : "
        f"{hermitian_error:.3e}"
    )

    print(
        f"Minimum eigenvalue : "
        f"{np.min(eigenvalues):.12e}"
    )

    print(
        f"CHSH S             : "
        f"{chsh_value(ideal):.12f}"
    )

    assert abs(
        trace.real - 1.0
    ) < 1e-12

    assert hermitian_error < 1e-12

    assert np.min(
        eigenvalues
    ) > -1e-12

    print("Status             : PASS")

    # ------------------------------------------------------------------
    # 2. Werner state
    # ------------------------------------------------------------------

    print()
    print("2. Werner source")
    print("-" * 70)

    rho = werner_state(
        E91_VISIBILITY
    )

    print(
        f"Visibility         : "
        f"{E91_VISIBILITY:.6f}"
    )

    print(
        f"Trace              : "
        f"{np.trace(rho).real:.12f}"
    )

    print(
        f"CHSH S             : "
        f"{chsh_value(rho):.12f}"
    )

    expected_s = (
        E91_VISIBILITY
        * 2.0
        * np.sqrt(2.0)
    )

    print(
        f"Expected S         : "
        f"{expected_s:.12f}"
    )

    print(
        f"Difference         : "
        f"{chsh_value(rho) - expected_s:+.3e}"
    )

    assert np.isclose(
        chsh_value(rho),
        expected_s,
        atol=1e-10,
    )

    print("Status             : PASS")

    # ------------------------------------------------------------------
    # 3. All nine basis correlations
    # ------------------------------------------------------------------

    print()
    print("3. Basis correlation audit")
    print("-" * 70)

    print_correlations(rho)

    # ------------------------------------------------------------------
    # 4. Noise-channel physicality
    # ------------------------------------------------------------------

    print()
    print("4. Noise-channel physicality audit")
    print("-" * 70)

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        noisy = apply_noise(
            rho,
            0.05,
            scenario,
        )

        trace = np.trace(
            noisy
        ).real

        eigenvalues = np.linalg.eigvalsh(
            noisy
        )

        print(
            f"{scenario:>7} | "
            f"trace={trace:.12f} | "
            f"min_eigenvalue="
            f"{np.min(eigenvalues):+.3e} | "
            f"CHSH={chsh_value(noisy):.6f}"
        )

        assert np.isclose(
            trace,
            1.0,
            atol=1e-10,
        )

        assert np.min(
            eigenvalues
        ) >= -1e-10

    print("Status             : PASS")

    # ------------------------------------------------------------------
    # 5. Exact CHSH noise sweep
    # ------------------------------------------------------------------

    print()
    print("5. Exact CHSH noise sweep")
    print("-" * 70)

    probabilities = np.arange(
        0.01,
        0.101,
        0.01,
    )

    print(
        f"{'scenario':<9}"
        f"{'p':>7}"
        f"{'QBER':>13}"
        f"{'CHSH S':>13}"
    )

    print("-" * 50)

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        for p in probabilities:

            result = simulate_e91(
                num_pairs=100_000,
                p=float(p),
                scenario=scenario,
                visibility=E91_VISIBILITY,
                seed=20260817,
            )

            print(
                f"{scenario:<9}"
                f"{p:>7.2f}"
                f"{result.qber:>13.6f}"
                f"{result.chsh_s:>13.6f}"
            )

    # ------------------------------------------------------------------
    # 6. Existing protocol convergence check
    # ------------------------------------------------------------------

    print()
    print("6. Existing protocol convergence check")
    print("-" * 70)

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        exact_rho = apply_noise(
            rho,
            0.05,
            scenario,
        )

        exact_s = chsh_value(
            exact_rho
        )

        result = simulate_e91(
            num_pairs=100_000,
            p=0.05,
            scenario=scenario,
            visibility=E91_VISIBILITY,
            seed=20260817,
        )

        print(
            f"{scenario:<9} | "
            f"exact S={exact_s:.6f} | "
            f"protocol S={result.chsh_s:.6f}"
        )

    # ------------------------------------------------------------------
    # 7. Independent Monte-Carlo CHSH check
    # ------------------------------------------------------------------

    print()
    print("7. Independent Monte-Carlo CHSH check")
    print("-" * 70)

    for scenario in [
        "single",
        "dual",
        "triple",
    ]:

        noisy_rho = apply_noise(
            rho,
            0.05,
            scenario,
        )

        exact_s = chsh_value(
            noisy_rho
        )

        mc_s, correlations_mc = monte_carlo_chsh(
            noisy_rho,
            CHSH_SETTINGS,
            samples_per_setting=10_000,
            seed=20260817,
        )

        difference = (
            mc_s - exact_s
        )

        print(
            f"{scenario:<9} | "
            f"exact S={exact_s:.6f} | "
            f"MC S={mc_s:.6f} | "
            f"diff={difference:+.6f}"
        )

        print(
            f"           "
            f"Eab={correlations_mc['ab']:+.6f} | "
            f"Eab'={correlations_mc['ab_prime']:+.6f} | "
            f"Ea'b={correlations_mc['a_prime_b']:+.6f} | "
            f"Ea'b'={correlations_mc['a_prime_b_prime']:+.6f}"
        )

    print()
    print("Audit complete.")


if __name__ == "__main__":
    main() 