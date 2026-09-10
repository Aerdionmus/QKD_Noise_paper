# Phase 2 implementation decisions

## Frozen now

### Noise channels

The implementation follows the manuscript:
- Depolarizing: `(1-p)ρ + p/3(XρX + YρY + ZρZ)`
- Dephasing: `(1-p)ρ + p ZρZ`
- Amplitude damping: `K0ρK0† + K1ρK1†`

Triple ordering:
`N_AD ∘ N_deph ∘ N_dep`, with `γ = p`.

The frozen ordering analysis covers all six permutations of `D`, `P`, and
`A`. `D` commutes with `P`, `P` commutes with `A`, and `D` does not commute
with `A`. The six permutations collapse into exactly two channel classes:

- Class 1: `A∘P∘D`, `A∘D∘P`, `P∘A∘D`
- Class 2: `P∘D∘A`, `D∘A∘P`, `D∘P∘A`

The classes have a common linear Bloch map and differ by
`Delta t_z = 4 d gamma / 3`; under the shared convention
`d=q=gamma=p`, this is `4 p^2 / 3`.

### QBER

The reference engine calculates state-level QBER directly from density matrices.

The standard symmetric four-state BB84 average QBER is ordering-independent.
The conditioned Z-basis quantities remain ordering-sensitive:

`Q0 = (1-Mz-tz)/2`, `Q1 = (1-Mz+tz)/2`, and `Q0-Q1 = -tz`.

For the two-sided Bell-state study:

`Tzz = Mz^2 + tz^2`,
`Delta Tzz = 8 d gamma^2 (3-2d) / 9`,
and `C = max(0, Mx^2 - (1-Tzz)/2)`.

The operational CHSH quantity is the fixed-setting value
`S_fixed = sqrt(2) (Txx + Tzz)`. The separately reported Horodecki optimum
is diagnostic only.

The channel-discrimination result is
`||Lambda_1-Lambda_2||_diamond = 4 d gamma / 3` and
`P_success = 1/2 + d gamma / 3`.

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

The legacy protocol implementation retains its own CHSH-setting convention.
The separate ordering study uses the explicitly fixed observable `S_fixed` and
must not be conflated with the legacy protocol simulation. The DIQKD rate in
the ordering study applies an established literature bound only when
`S_fixed >= 2`; it is not a newly proved security theorem.

### Werner visibility

The frozen robustness model is
`rho_W(v) = v |Phi+><Phi+| + (1-v) I/4`.
Its validated two-sided correlation tensor is
`T(v) = diag(v Mx^2, -v Mx^2, v Mz^2 + tz^2)`.
Thus `Delta Tzz` is exactly independent of `v` for this model, although
decreasing `v` shrinks the operational regions for concurrence, CHSH
violation, and DIQKD positivity.

### Dark-count expression

The literal Eq. (13) is exposed as a diagnostic because α is reported in dB/km.

### Super-additivity

The stated Kraus channels are implemented literally. No artificial cross-term is added to force agreement with the paper's theorem.

The repository does not treat “sub-additive quadratic correction” or
“super-additivity” as validated results. The frozen contribution is the exact
ordering classification, its affine defect, and the operational consequences
listed above.
