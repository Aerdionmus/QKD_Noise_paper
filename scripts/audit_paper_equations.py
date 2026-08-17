import math


# ============================================================
# Paper parameters — Table 1 / Section V-E
# ============================================================

P = 0.05

ETA = 0.15
DARK_COUNT_RATE = 5e-6

MU = 0.1

DISTANCE_KM = 50.0
ATTENUATION_DB_PER_KM = 0.2

MISALIGNMENT_DEG = 2.0


# ============================================================
# Paper equations
# ============================================================

def q_align(epsilon_deg: float) -> float:
    """
    Paper Eq. / Section V-E:

        Q_align = sin^2(epsilon)

    epsilon = 2 degrees.
    """
    epsilon_rad = math.radians(epsilon_deg)
    return math.sin(epsilon_rad) ** 2


def fiber_transmittance_db(
    distance_km: float,
    attenuation_db_per_km: float,
) -> float:
    """
    Physical fiber power transmittance corresponding to:

        alpha = 0.2 dB/km
        L = 50 km

        T = 10^(-alpha L / 10)

    NOTE:
    This expression is NOT explicitly written in the paper's
    Eq. (13), but is the standard conversion from dB loss
    to power transmittance.
    """
    return 10 ** (
        -attenuation_db_per_km * distance_km / 10.0
    )


def q_dark_using_db_transmittance() -> float:
    """
    Interpret Eq. (13) using the physical dB transmittance:

        Q_dark = d / (2 eta mu T)

    where

        T = 10^(-alpha L / 10)
    """
    T = fiber_transmittance_db(
        DISTANCE_KM,
        ATTENUATION_DB_PER_KM,
    )

    return DARK_COUNT_RATE / (
        2.0 * ETA * MU * T
    )


def q_dark_literal_paper_equation() -> float:
    """
    Literal transcription of Eq. (13):

        Q_eff = Q_noise + d / (2 eta mu e^(-alpha L))

    IMPORTANT:
    The paper gives alpha in dB/km, but Eq. (13) uses
    exp(-alpha L). These are dimensionally inconsistent.

    We calculate it literally so that the discrepancy is
    documented rather than silently corrected.
    """
    attenuation_exponent = math.exp(
        -ATTENUATION_DB_PER_KM * DISTANCE_KM
    )

    return DARK_COUNT_RATE / (
        2.0 * ETA * MU * attenuation_exponent
    )


def paper_reported_dark_contribution() -> float:
    """
    Section V-E states that the dark-count contribution
    adds approximately 0.003 to the base QBER.
    """
    return 0.003


def paper_reported_q0() -> float:
    """
    Section V-E states:

        Q0 approximately 0.006

    This is reported by the paper but is not fully derivable
    from the preceding equations as written.
    """
    return 0.006


def binary_entropy(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        if q == 0.0 or q == 1.0:
            return 0.0
        raise ValueError("q must be in [0, 1]")

    return (
        -q * math.log2(q)
        - (1.0 - q) * math.log2(1.0 - q)
    )


def secret_key_rate_paper(q: float) -> float:
    """
    Paper Eq. (19):

        R = max(0, 1 - 2H(Q))
            * (1 - mu/2)
            - Delta_PNS
    """
    h = binary_entropy(q)

    delta_pns = MU * math.exp(
        -ATTENUATION_DB_PER_KM * DISTANCE_KM
    )

    return (
        max(0.0, 1.0 - 2.0 * h)
        * (1.0 - MU / 2.0)
        - delta_pns
    )


def secret_key_rate_using_db_transmittance(q: float) -> float:
    """
    Alternative physically consistent interpretation:

        Delta_PNS = mu * T

    where T is the dB-derived fiber transmittance.

    This is reported only for comparison.
    """
    h = binary_entropy(q)

    T = fiber_transmittance_db(
        DISTANCE_KM,
        ATTENUATION_DB_PER_KM,
    )

    delta_pns = MU * T

    return (
        max(0.0, 1.0 - 2.0 * h)
        * (1.0 - MU / 2.0)
        - delta_pns
    )


def pri(rate: float, cumulative_noise: float) -> float:
    """
    Paper Eq. (20):

        PRI = R(Sigma_p) / Sigma_p
    """
    if cumulative_noise <= 0.0:
        return float("inf")

    return rate / cumulative_noise


# ============================================================
# Main audit
# ============================================================

def main():
    print("Paper Equation Audit — Track A")
    print("================================")
    print()

    print("Parameters")
    print(f"p                     : {P}")
    print(f"eta                   : {ETA}")
    print(f"dark count d          : {DARK_COUNT_RATE:.6e}")
    print(f"mu                    : {MU}")
    print(f"distance              : {DISTANCE_KM} km")
    print(
        f"attenuation           : "
        f"{ATTENUATION_DB_PER_KM} dB/km"
    )
    print(
        f"misalignment          : "
        f"{MISALIGNMENT_DEG} degrees"
    )
    print()

    # --------------------------------------------------------
    # Misalignment
    # --------------------------------------------------------

    qalign = q_align(MISALIGNMENT_DEG)

    print("1. BASIS MISALIGNMENT")
    print("----------------------")
    print(f"Calculated Q_align    : {qalign:.8f}")
    print(
        f"Paper reported        : "
        f"approximately 0.0012"
    )
    print(
        f"Difference from 0.0012: "
        f"{qalign - 0.0012:+.8f}"
    )
    print()

    # --------------------------------------------------------
    # Fiber transmittance
    # --------------------------------------------------------

    T = fiber_transmittance_db(
        DISTANCE_KM,
        ATTENUATION_DB_PER_KM,
    )

    print("2. FIBER TRANSMITTANCE")
    print("----------------------")
    print(f"T = 10^(-alpha L/10)  : {T:.8f}")
    print()

    # --------------------------------------------------------
    # Dark count
    # --------------------------------------------------------

    qdark_db = q_dark_using_db_transmittance()
    qdark_literal = q_dark_literal_paper_equation()

    print("3. DARK-COUNT CONTRIBUTION")
    print("---------------------------")

    print(
        "Physical dB interpretation:"
    )
    print(
        f"Q_dark                : "
        f"{qdark_db:.8f}"
    )

    print()
    print(
        "Literal Eq. (13) interpretation:"
    )
    print(
        f"Q_dark                : "
        f"{qdark_literal:.8f}"
    )

    print()
    print(
        "Paper's stated numerical claim:"
    )
    print(
        f"Reported contribution  : "
        f"{paper_reported_dark_contribution():.8f}"
    )

    print()

    # --------------------------------------------------------
    # Q0
    # --------------------------------------------------------

    q0_from_align_and_db_dark = (
        qalign + qdark_db
    )

    print("4. PRACTICAL QBER FLOOR")
    print("------------------------")

    print(
        f"Q_align + Q_dark      : "
        f"{q0_from_align_and_db_dark:.8f}"
    )

    print(
        f"Paper reported Q0     : "
        f"{paper_reported_q0():.8f}"
    )

    print(
        f"Difference             : "
        f"{q0_from_align_and_db_dark - paper_reported_q0():+.8f}"
    )

    print()
    print(
        "IMPORTANT: The paper reports Q0 ≈ 0.006,"
    )
    print(
        "but the explicitly stated alignment and dark-count"
    )
    print(
        "equations do not uniquely reproduce 0.006."
    )

    print()

    # --------------------------------------------------------
    # PNS
    # --------------------------------------------------------

    pns_literal = (
        MU
        * math.exp(
            -ATTENUATION_DB_PER_KM
            * DISTANCE_KM
        )
    )

    pns_db = (
        MU
        * T
    )

    print("5. PNS KEY-RATE PENALTY")
    print("-----------------------")

    print(
        "Literal paper expression:"
    )
    print(
        f"Delta_PNS             : "
        f"{pns_literal:.8f}"
    )

    print()
    print(
        "Using physical dB transmittance:"
    )
    print(
        f"Delta_PNS             : "
        f"{pns_db:.8f}"
    )

    print()

    # --------------------------------------------------------
    # Key rate comparison
    # --------------------------------------------------------

    test_q = 0.05

    rate_literal = secret_key_rate_paper(test_q)

    rate_db = (
        secret_key_rate_using_db_transmittance(
            test_q
        )
    )

    print("6. SECRET KEY RATE")
    print("------------------")
    print(f"Test QBER             : {test_q:.6f}")
    print(
        f"Paper-literal R       : "
        f"{rate_literal:.8f}"
    )
    print(
        f"Physical-T R          : "
        f"{rate_db:.8f}"
    )

    print()

    # --------------------------------------------------------
    # PRI
    # --------------------------------------------------------

    print("7. PRI")
    print("------")

    for scenario, channels in [
        ("single", 1),
        ("dual", 2),
        ("triple", 3),
    ]:
        sigma_p = channels * P

        rate = secret_key_rate_paper(
            test_q
        )

        value = pri(
            rate,
            sigma_p,
        )

        print(
            f"{scenario:<8} "
            f"Sigma_p={sigma_p:.4f} "
            f"PRI={value:.8f}"
        )

    print()

    # --------------------------------------------------------
    # Final assessment
    # --------------------------------------------------------

    print("TRACK-A ASSESSMENT")
    print("==================")
    print()
    print("MATCH:")
    print(
        "  - Q_align equation is numerically consistent."
    )
    print(
        "  - Noise-channel definitions are explicitly stated."
    )
    print(
        "  - Key-rate and PRI equations are explicitly stated."
    )
    print()
    print("REQUIRES DOCUMENTATION:")
    print(
        "  - Eq. (13) mixes dB/km alpha with exp(-alpha L)."
    )
    print(
        "  - Paper reports dark contribution ~0.003,"
    )
    print(
        "    which does not directly follow from Eq. (13)"
    )
    print(
        "    under the stated parameters."
    )
    print(
        "  - Paper reports Q0 ~= 0.006 without a complete"
    )
    print(
        "    derivation from the preceding practical terms."
    )
    print()
    print(
        "Therefore Track A will preserve the paper's"
    )
    print(
        "equations/claims and explicitly flag inconsistencies."
    )


if __name__ == "__main__":
    main()
