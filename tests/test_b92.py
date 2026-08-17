import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qkd_noise.protocols.b92 import (
    KET_0,
    KET_PLUS,
    density_matrix,
    apply_noise,
    z_measurement_probabilities,
    x_measurement_probabilities,
    simulate_transmission,
    simulate_b92,
)


SCENARIOS = [
    "single",
    "dual",
    "triple",
]


def exact_b92(p, scenario):
    """
    Exact B92 reference calculation.

    Alice:
        0 -> |0>
        1 -> |+>

    Bob:
        Z:
            outcome |1> is conclusive -> infer 1

        X:
            outcome |-> is conclusive -> infer 0

    QBER is conditioned on conclusive events.
    """

    states = {
        0: KET_0,
        1: KET_PLUS,
    }

    total_conclusive = 0.0
    total_errors = 0.0

    for alice_bit in (0, 1):

        rho = density_matrix(
            states[alice_bit]
        )

        rho = apply_noise(
            rho,
            p,
            scenario,
        )

        # Bob chooses Z with probability 1/2.
        p0, p1 = z_measurement_probabilities(rho)

        total_conclusive += 0.25 * p1

        if alice_bit == 0:
            total_errors += 0.25 * p1

        # Bob chooses X with probability 1/2.
        p_plus, p_minus = x_measurement_probabilities(rho)

        total_conclusive += 0.25 * p_minus

        if alice_bit == 1:
            total_errors += 0.25 * p_minus

    qber = (
        total_errors / total_conclusive
        if total_conclusive > 0.0
        else 0.0
    )

    return {
        "qber": qber,
        "conclusive_probability": total_conclusive,
        "error_probability": total_errors,
    }


def test_exact_b92_has_valid_probabilities():
    for scenario in SCENARIOS:
        for p in np.linspace(0.01, 0.10, 10):

            result = exact_b92(
                p,
                scenario,
            )

            assert 0.0 <= result["qber"] <= 1.0
            assert 0.0 < result["conclusive_probability"] <= 1.0
            assert 0.0 <= result["error_probability"] <= 1.0


def test_exact_b92_noise_ordering():
    for p in np.linspace(0.01, 0.10, 10):

        single = exact_b92(
            p,
            "single",
        )["qber"]

        dual = exact_b92(
            p,
            "dual",
        )["qber"]

        triple = exact_b92(
            p,
            "triple",
        )["qber"]

        assert single < dual
        assert dual < triple


def test_zero_noise_b92():
    for scenario in SCENARIOS:

        result = simulate_b92(
            n=20_000,
            p=0.0,
            scenario=scenario,
            seed=12345,
        )

        # At p=0, the only errors are intrinsic to the
        # non-orthogonal B92 states / conclusive measurement
        # structure. The simulation must remain well-defined.
        assert 0.0 <= result["qber"] <= 1.0
        assert result["conclusive_count"] > 0


def test_simulate_b92_is_reproducible():

    result_a = simulate_b92(
        n=5000,
        p=0.05,
        scenario="dual",
        seed=12345,
    )

    result_b = simulate_b92(
        n=5000,
        p=0.05,
        scenario="dual",
        seed=12345,
    )

    assert result_a["qber"] == result_b["qber"]
    assert (
        result_a["conclusive_count"]
        == result_b["conclusive_count"]
    )


def test_simulate_transmission_is_valid():

    rng = np.random.default_rng(12345)

    for scenario in SCENARIOS:
        for alice_bit in (0, 1):
            for bob_basis in (0, 1):

                result = simulate_transmission(
                    alice_bit=alice_bit,
                    bob_basis=bob_basis,
                    p=0.05,
                    scenario=scenario,
                    rng=rng,
                )

                assert "conclusive" in result
                assert "bob_bit" in result

                assert isinstance(
                    result["conclusive"],
                    (bool, np.bool_),
                )

                if result["conclusive"]:
                    assert result["bob_bit"] in (0, 1)
                else:
                    assert result["bob_bit"] is None
