
## Reproduction layers

The repository distinguishes five kinds of evidence:

1. **Exact analytic results:** affine channel maps, pairwise commutation
   relations, the two six-order equivalence classes, the translation defect,
   BB84 conditioned formulas, two-sided Bell correlations, concurrence,
   fixed-setting CHSH, and channel discrimination.
2. **Numerical validation/regression:** direct Kraus application, optional
   Qiskit SuperOp comparisons, generated CSVs, and the 90 automated tests.
3. **Numerical threshold results:** root-found concurrence, fixed-CHSH, and
   DIQKD-domain crossings reported by the E91 ordering and Werner studies.
4. **Established-bound application:** the DIQKD expression is evaluated only
   for `S_fixed >= 2`; it is an application of the cited literature bound, not
   a new security theorem.
5. **Legacy protocol simulations:** BB84/B92/E91 event-level simulations and
   cross-protocol summaries remain separate from the exact ordering studies.

The Werner robustness study uses
`rho_W(v) = v|Phi+><Phi+| + (1-v)I/4` and validates
`T(v)=diag(v Mx^2, -v Mx^2, v Mz^2 + tz^2)`. Its key result is that
`Delta Tzz` is exactly independent of `v`, while reduced visibility shrinks
the absolute concurrence, CHSH-violating, and DIQKD-positive regions.

## Protocol-level implementation status

The protocol-level reproduction stage is now complete for:

- BB84
- B92
- E91

Each protocol was evaluated under:

- single noise
- dual noise
- triple noise

for `p = 0.01...0.10`.

### Statistical treatment

The final protocol simulations use 50 independent trials per data point.

QBER confidence intervals are reported for the final protocol summaries.

The cross-protocol comparison is performed using the corresponding final summary datasets rather than mixing intermediate/raw datasets.

### Cross-protocol comparison

The cross-protocol analysis is descriptive and restricted to the implemented model and parameter range.

It does not establish universal protocol superiority.

BB84 has the lowest simulated QBER throughout the tested parameter sweep. B92 generally has the highest QBER at moderate and high noise. E91 additionally provides CHSH as a nonlocality diagnostic.

### E91 interpretation

CHSH is treated separately from QBER.

A CHSH value below 2 indicates that the simulated correlations no longer violate the classical Bell bound under the implemented model. This is not treated as equivalent to a QBER security threshold.

The legacy E91 simulation uses its own executable measurement convention. The
new ordering studies instead use the explicitly defined fixed-setting
observable `S_fixed`; reconciliation of the two implementations is intentionally
outside the frozen scope.

### Security metrics

Protocol-specific security metrics are not assumed to be directly comparable unless their definitions are consistent.

In particular:

- BB84 secret-key rate follows the frozen Eq. (19) implementation.
- PRI follows the literal Eq. (20) implementation.
- E91 secure-fraction output is retained as an implementation metric.
- B92 does not receive a fabricated secret-key-rate metric where the final summary does not define one.

### Reproduction principle

Where the manuscript is ambiguous, the implementation records the chosen interpretation and reports the resulting numerical behavior.

No result is modified solely to force agreement with a reported manuscript value.

## Frozen ordering result

`D` and `P` commute, `P` and `A` commute, and `D` and `A` do not commute.
All six permutations collapse into:

- Class 1: `A o P o D`, `A o D o P`, `P o A o D`
- Class 2: `P o D o A`, `D o A o P`, `D o P o A`

The classes share the same linear map and have
`Delta t_z = 4 d gamma / 3`; for `d=q=gamma=p`, this becomes
`Delta t_z = 4 p^2 / 3`.

The symmetric BB84 average QBER is ordering-independent, whereas
`Q0-Q1=-tz`, `Tzz=Mz^2+tz^2`, concurrence, and fixed-setting CHSH expose
the ordering defect. Channel discrimination is
`||Lambda_1-Lambda_2||_diamond = 4 d gamma / 3` with
`P_success = 1/2 + d gamma / 3`.

## Retained limitations

The following remain unresolved and are not silently repaired:

- B92 conclusive-outcome mapping is incomplete.
- PRI values remain inconsistent with the manuscript's Eq. (20).
- The dark-count attenuation expression is inconsistent with the stated dB/km
  parameterization and is retained only as an audit diagnostic.
- The manuscript's stated super-additivity theorem is not reproduced by the
  literal Kraus maps.
- Legacy E91 setting/interpretation issues remain separate from the fixed-setting
  ordering study.
