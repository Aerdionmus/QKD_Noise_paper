from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    """Parameters transcribed from the manuscript."""

    n_qubits: int = 1000
    trials: int = 50
    bootstrap_resamples: int = 1000

    detector_efficiency: float = 0.15
    dark_count_rate: float = 5e-6
    mean_photon_number: float = 0.1

    misalignment_deg: float = 2.0
    fiber_distance_km: float = 50.0
    attenuation_db_per_km: float = 0.2

    e91_visibility: float = 0.97

    bb84_threshold: float = 0.11
    b92_threshold: float = 0.06
    e91_threshold: float = 0.146

    @property
    def q_alignment(self) -> float:
        import math
        return math.sin(math.radians(self.misalignment_deg)) ** 2

    @property
    def delta_pns(self) -> float:
        # Literal manuscript expression: μ exp(-αL).
        import math
        return self.mean_photon_number * math.exp(
            -self.attenuation_db_per_km * self.fiber_distance_km
        )

    @property
    def dark_count_qber_paper_literal(self) -> float:
        # Literal Eq. (13), intentionally exposed as a diagnostic.
        import math
        return self.dark_count_rate / (
            2
            * self.detector_efficiency
            * self.mean_photon_number
            * math.exp(-self.attenuation_db_per_km * self.fiber_distance_km)
        )

    @property
    def q0_reported(self) -> float:
        return 0.006
