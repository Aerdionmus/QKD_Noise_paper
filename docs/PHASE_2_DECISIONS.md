# Phase 2 implementation decisions

## Frozen now

### Noise channels

The implementation follows the manuscript:
- Depolarizing: `(1-p)ρ + p/3(XρX + YρY + ZρZ)`
- Dephasing: `(1-p)ρ + p ZρZ`
- Amplitude damping: `K0ρK0† + K1ρK1†`

Triple ordering:
`N_AD ∘ N_deph ∘ N_dep`, with `γ = p`.

### QBER

The reference engine calculates state-level QBER directly from density matrices.

Practical imperfections remain separate until protocol event simulation is implemented.

### Secret key rate

Eq. (19) is authoritative:
`R = max(0, 1 - 2H(Q)) * (1 - μ/2) - ΔPNS`

The Algorithm 1 variant is recorded as an inconsistency.

### PRI

Eq. (20) is implemented literally:
`PRI = R(Σp) / Σp`

## Not silently resolved

### B92

The paper gives `|0>` and `|+>` and mentions X/Z measurements and conclusive outcomes, but no complete conclusive-outcome table.

### E91

The paper gives angle sets and the CHSH equation but not the four explicit CHSH setting assignments.

### Dark-count expression

The literal Eq. (13) is exposed as a diagnostic because α is reported in dB/km.

### Super-additivity

The stated Kraus channels are implemented literally. No artificial cross-term is added to force agreement with the paper's theorem.
