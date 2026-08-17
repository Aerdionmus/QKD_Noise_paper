import math


def binary_entropy(q: float) -> float:
    if q < 0 or q > 1:
        raise ValueError("QBER must be in [0,1].")
    if q in (0.0, 1.0):
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def secret_key_rate(q: float, mu: float = 0.1, delta_pns: float = 0.0) -> float:
    """Paper Eq. (19)."""
    base = max(0.0, 1.0 - 2.0 * binary_entropy(q))
    return base * (1.0 - mu / 2.0) - delta_pns


def pri(rate: float, cumulative_noise: float) -> float:
    """Paper Eq. (20): PRI = R(Σp) / Σp."""
    if cumulative_noise <= 0:
        raise ValueError("cumulative_noise must be > 0.")
    return rate / cumulative_noise


def effective_qber(
    q_noise: float,
    *,
    dark_count_term: float = 0.0,
    misalignment_qber: float = 0.0,
) -> float:
    """Explicit practical-QBER post-processing helper."""
    return min(1.0, q_noise + dark_count_term + misalignment_qber)
