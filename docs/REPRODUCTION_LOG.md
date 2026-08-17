# Reproduction log

## Phase 2 — initial implementation

Source: uploaded manuscript *Composite Noise Interaction in Discrete-Variable QKD*.

Parameters:
- Qiskit Aer 0.13
- 1000 qubits/trial
- 50 trials/data point
- 1000 bootstrap resamples
- p = 0.01...0.10 in steps of 0.01
- η = 0.15
- d = 5e-6
- μ = 0.1
- ε = 2°
- L = 50 km
- α = 0.2 dB/km
- E91 visibility = 0.97

First validation target:
implement the Kraus layer independently before constructing QKD circuits.

For the stated depolarizing channel, the four BB84 states give ideal QBER `2p/3`.

The dual and triple cascades are then evaluated directly from the density-matrix composition.
