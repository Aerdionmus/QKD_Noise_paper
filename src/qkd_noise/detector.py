import math


def dark_count_probability(
    dark_count_rate: float,
) -> float:
    """
    Convert a dark-count rate/probability parameter into
    a per-gate dark-click probability.

    For the present model we treat the supplied dark-count
    parameter as the probability of a dark click in one
    detector gate.
    """

    if not 0.0 <= dark_count_rate <= 1.0:
        raise ValueError(
            "dark_count_rate must be between 0 and 1"
        )

    return dark_count_rate


def apply_dark_count_to_bit(
    ideal_bit: int,
    dark_count_rate: float,
    rng,
) -> tuple[int, bool]:
    """
    Simple detector dark-count model.

    If a dark count occurs, the detector output is modeled
    as a random binary result.
    """

    p_dark = dark_count_probability(
        dark_count_rate
    )

    dark_event = (
        rng.random() < p_dark
    )

    if not dark_event:
        return int(ideal_bit), False

    detected_bit = int(
        rng.integers(0, 2)
    )

    return detected_bit, True


def dark_count_qber_contribution(
    dark_count_rate: float,
) -> float:
    """
    QBER contribution of the simple symmetric dark-count
    model.

    A dark event produces a random bit, therefore its
    conditional error probability is 1/2.
    """

    return (
        dark_count_probability(
            dark_count_rate
        )
        * 0.5
    )


def channel_transmittance(
    distance_km: float,
    attenuation_db_per_km: float,
) -> float:
    """
    Fiber power transmittance:

        T = 10^(-alpha * L / 10)
    """

    if distance_km < 0:
        raise ValueError(
            "distance_km must be non-negative"
        )

    if attenuation_db_per_km < 0:
        raise ValueError(
            "attenuation_db_per_km must be non-negative"
        )

    return 10 ** (
        -attenuation_db_per_km
        * distance_km
        / 10.0
    )


def total_detection_probability(
    mean_photon_number: float,
    distance_km: float,
    attenuation_db_per_km: float,
    detector_efficiency: float,
) -> float:
    """
    Approximate probability of detecting at least one
    photon from a weak coherent pulse.

        T = 10^(-alpha * L / 10)

        P_signal = 1 - exp(-mu * T * eta)

    This is an explicit physical modeling choice for
    the reproduction study.
    """

    if mean_photon_number < 0:
        raise ValueError(
            "mean_photon_number must be non-negative"
        )

    if not 0.0 <= detector_efficiency <= 1.0:
        raise ValueError(
            "detector_efficiency must be between 0 and 1"
        )

    transmittance = channel_transmittance(
        distance_km,
        attenuation_db_per_km,
    )

    exponent = (
        -mean_photon_number
        * transmittance
        * detector_efficiency
    )

    return 1.0 - math.exp(exponent)


def detect_bit(
    ideal_bit: int,
    mean_photon_number: float,
    distance_km: float,
    attenuation_db_per_km: float,
    detector_efficiency: float,
    dark_count_rate: float,
    rng,
) -> tuple[int | None, str]:
    """
    Simplified WCP detector model.

    Returns
    -------
    detected_bit:
        0 or 1 if a detection occurs.
        None if no detection occurs.

    event_type:
        "signal" if a genuine signal is detected.
        "dark"   if only a dark count is detected.
        "none"   if no detection occurs.

    Model
    -----
    P_signal = 1 - exp(-mu * T * eta)

    If a signal is detected:
        retain the ideal bit.

    If no signal is detected:
        a dark count may produce a random bit.

    This is an explicit physical modeling choice for
    the reproduction study. It is not claimed to be the
    exact detector implementation used by the manuscript.
    """

    if not 0.0 <= dark_count_rate <= 1.0:
        raise ValueError(
            "dark_count_rate must be between 0 and 1"
        )

    p_signal = total_detection_probability(
        mean_photon_number=mean_photon_number,
        distance_km=distance_km,
        attenuation_db_per_km=attenuation_db_per_km,
        detector_efficiency=detector_efficiency,
    )

    # Genuine signal detection.
    if rng.random() < p_signal:
        return int(ideal_bit), "signal"

    # No genuine signal detected.
    # A dark count can create a random detector output.
    if rng.random() < dark_count_rate:
        random_bit = int(
            rng.integers(0, 2)
        )

        return random_bit, "dark"

    # No detector event.
    return None, "none"