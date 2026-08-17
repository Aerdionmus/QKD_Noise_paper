import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _check_probability(p: float):
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {p}")


def depolarizing_kraus(p: float):
    """Paper Eq. (3)."""
    _check_probability(p)
    return [
        np.sqrt(1 - p) * I,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def dephasing_kraus(p: float):
    """Paper Eq. (4)."""
    _check_probability(p)
    return [
        np.sqrt(1 - p) * I,
        np.sqrt(p) * Z,
    ]


def amplitude_damping_kraus(gamma: float):
    """Paper Eq. (6)."""
    _check_probability(gamma)
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def apply_channel(rho: np.ndarray, kraus_ops) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for k in kraus_ops:
        out += k @ rho @ k.conj().T
    return out


def compose_channels(rho: np.ndarray, *kraus_channels) -> np.ndarray:
    """Apply channels left-to-right: N2(N1(rho)) for (N1, N2)."""
    out = rho.astype(complex, copy=True)
    for channel in kraus_channels:
        out = apply_channel(out, channel)
    return out


def bb84_states():
    return {
        0: np.array([1, 0], dtype=complex),
        1: np.array([0, 1], dtype=complex),
        2: np.array([1, 1], dtype=complex) / np.sqrt(2),
        3: np.array([1, -1], dtype=complex) / np.sqrt(2),
    }


def projector(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def bb84_state_qber(rho: np.ndarray, state_id: int) -> float:
    states = bb84_states()
    if state_id in (0, 1):
        wrong = states[1 - state_id]
    else:
        wrong = states[5 - state_id]
    return float(np.real(wrong.conj() @ rho @ wrong))


def bb84_average_qber(p: float, scenario: str = "single") -> float:
    """Exact BB84 average QBER from the stated Kraus cascade.

    Practical imperfections are deliberately not included here.
    """
    if scenario not in {"single", "dual", "triple"}:
        raise ValueError("scenario must be single, dual, or triple")

    channels = [depolarizing_kraus(p)]
    if scenario in {"dual", "triple"}:
        channels.append(dephasing_kraus(p))
    if scenario == "triple":
        channels.append(amplitude_damping_kraus(p))

    qs = []
    for state_id, psi in bb84_states().items():
        rho = compose_channels(projector(psi), *channels)
        qs.append(bb84_state_qber(rho, state_id))
    return float(np.mean(qs))
