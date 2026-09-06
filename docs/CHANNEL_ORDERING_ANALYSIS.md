# Channel Ordering Analysis: D, P, A Composite Noise

## Status of this document

This document distinguishes four kinds of statement throughout:

- **EXACT RESULT** — proved algebraically from the Kraus operators in
  `src/qkd_noise/channels.py`, valid for all `p in [0,1]` (with `gamma = p`
  for amplitude damping, matching the repository's existing convention).
- **NUMERICAL RESULT** — a finite-precision check (`TOL = 1e-12`) that
  confirms an exact result on a representative sample; it is evidence,
  not proof, but is reported because it independently corroborates the
  algebra using a completely different code path (direct Kraus application
  and, where available, Qiskit `SuperOp`).
- **INTERPRETATION** — a reading of what the exact/numerical results mean
  for the paper; not itself a new mathematical claim.
- **OPEN QUESTION** — anything this analysis does not settle.

No canonical channel definition, parameter convention, or existing
analytical result in the repository is modified by this document or by
the accompanying code. `src/qkd_noise/channels.py` is unmodified.

## 1. Motivation

The repository's canonical composite-noise pipeline is the triple ordering

```
N_AD ∘ N_deph ∘ N_dep      (gamma = p)
```

i.e. "A o P o D" in the notation used throughout this project
(`X o Y := "apply Y first, then X"`, matching the convention already
established in `scripts/audit_channel_commutation_dp.py`). The existing
`docs/PHASE_2_DECISIONS.md` freezes this specific ordering. Before
expanding the audit to the full space of orderings, this document asks:
does the *ordering* of D (depolarizing), P (dephasing), and A (amplitude
damping) matter, and if so, at which level — the channel itself, or the
BB84 observable computed from it?

## 2. Canonical channel definitions (by reference)

All definitions below are read directly from `src/qkd_noise/channels.py`
and are not altered:

- `depolarizing_kraus(p)`: Paper Eq. (3), Kraus set
  `{sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z}`.
- `dephasing_kraus(p)`: Paper Eq. (4), Kraus set
  `{sqrt(1-p) I, sqrt(p) Z}`.
- `amplitude_damping_kraus(gamma)`: Paper Eq. (6), standard amplitude-damping
  Kraus pair, with `gamma = p` used throughout this analysis (matching
  `bb84_average_qber`'s existing `scenario="triple"` convention).

## 3. Bloch-vector (affine) derivations — EXACT RESULT

Write a single-qubit state as `rho = (I + r . sigma) / 2` with Bloch
vector `r = (x, y, z)`. Every channel here is affine: `r' = M r + t`.

**Depolarizing.** Using the Pauli identity
`X rho X + Y rho Y + Z rho Z = (3I - r.sigma)/2` (direct computation from
conjugation rules `X sigma X = (+X,-Y,-Z)` etc.), substituting into
`(1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)` gives

```
M_D = (1 - 4p/3) * I_3        t_D = 0
```

`M_D` is a *scalar multiple of the identity matrix* — this single fact
drives most of the results below, since a scalar matrix commutes with
every other 3x3 matrix.

**Dephasing.** `(1-p) rho + p * Z rho Z` gives, by the same conjugation
rules,

```
M_P = diag(1-2p, 1-2p, 1)      t_P = 0
```

Note the exact `1` in the z,z entry: dephasing leaves the z-Bloch
component completely untouched.

**Amplitude damping.** From `K0 = diag(1, sqrt(1-gamma))`,
`K1 = [[0, sqrt(gamma)],[0,0]]`, direct Kraus application gives the
standard result

```
M_A = diag(sqrt(1-gamma), sqrt(1-gamma), 1-gamma)      t_A = (0, 0, gamma)
```

Amplitude damping is the *only* one of the three channels that is
non-unital (`t_A != 0`), and its translation lies entirely along z.

**NUMERICAL RESULT.** All three affine maps were reconstructed purely
numerically (probing each channel with `r in {0, e_x, e_y, e_z}` and
solving for `M, t`) at `p = 0.13` and found to agree with the closed-form
expressions above to machine precision (see
`scripts/audit_full_channel_orderings.py`, function
`_numeric_affine_map`, used only as an independent cross-check, not as
the basis for any claim in this document).

## 4. Pairwise commutation — EXACT RESULT

For two affine maps `f(r) = M_f r + t_f` and `g(r) = M_g r + t_g`,

```
(f o g)(r) - (g o f)(r) = (M_f M_g - M_g M_f) r + (M_f t_g + t_f) - (M_g t_f + t_g)
```

**D and P.** `M_D` is a scalar multiple of the identity, so
`M_D M_P = M_P M_D` for *any* `M_P` — the linear parts always commute.
Both `t_D = t_P = 0`, so the translation term vanishes identically. Hence

```
D o P = P o D   exactly, for all p in [0,1].
```

This matches (and gives an exact algebraic proof of) the existing
numerical finding in `results/ordering_audit/dp_channel_commutation*.csv`
(max Frobenius difference `~1.7e-16`, consistent with floating-point
noise around an exact zero). **This confirms Hypothesis 1** without
reinterpreting the existing audit.

**P and A.** `M_P` and `M_A` are both diagonal, so `M_P M_A = M_A M_P`
(diagonal matrices always commute elementwise, regardless of their
entries). For the translation term: `t_A = (0,0,gamma)` lies purely along
z, and `M_P`'s z,z entry is exactly `1`, so `M_P t_A = t_A`. Substituting,
both `P o A` and `A o P` reduce to the same map:

```
M_{PA} r + t_A  =  M_{AP} r + t_A
```

so

```
P o A = A o P   exactly, for all p in [0,1].
```

**This confirms Hypothesis 2.** The mechanism is specific: it relies on
dephasing preserving z exactly *and* amplitude damping's non-unital part
lying entirely along z. It is not a generic fact about arbitrary pairs of
Pauli/amplitude-damping-like channels.

**D and A.** `M_D M_A = M_A M_D = c_D M_A` (scalar commutes), so the
*linear* parts still agree. The translation parts do not:

```
(A o D)(r) = M_A M_D r + t_A          [D applied first]
(D o A)(r) = M_D M_A r + M_D t_A = c_D M_A r + c_D t_A   [A applied first]
```

with `c_D = 1 - 4p/3`. The difference is entirely in the z-translation:

```
t_A - c_D t_A = (1 - c_D) t_A = (4p/3) * (0, 0, gamma) = (0, 0, 4p^2/3)   [gamma = p]
```

which is nonzero for every `p in (0, 1]`. Only the **z-component** of the
Bloch vector differs; the linear (contraction) part is identical either
way.

```
D o A != A o D   for any p > 0. (They agree only at p = 0.)
```

**This confirms Hypothesis 3.** Numerically (see Section 7), the maximum
Frobenius discrepancy at `p = 0.13` is `1.593347e-02`, matching the
closed-form prediction `(4p^2/3)/2 * sqrt(2) = 1.593347e-02` for a
pure-z Bloch difference of that magnitude, to 6 significant figures.

## 5. Triple-ordering equivalence classes — EXACT RESULT

Write a time-ordered application `(X1, X2, X3)` (X1 applied first) for
composite map `X3 o X2 o X1`. For all six permutations of `{D, P, A}`:

- The **linear part** `M = M_{X3} M_{X2} M_{X1}` is the *same diagonal
  matrix* in all six orderings, because it is a product of `M_D`
  (scalar), `M_P` (diagonal) and `M_A` (diagonal) in some order, and
  diagonal matrices (including scalar matrices) always commute with each
  other, giving

  ```
  M = c_D * diag( sqrt(1-p)(1-2p), sqrt(1-p)(1-2p), 1-p )   for every ordering.
  ```

- The **translation part** is `t = M_{X3} M_{X2} t_{X1} + M_{X3} t_{X2} + t_{X3}`.
  Since `t_D = t_P = 0` and only `t_A = (0,0,p)` is nonzero, every term in
  this sum vanishes except the one contributed by A, weighted by the
  product of the M's of whichever channels are applied **after** A in
  time. Because `M_P t_A = t_A` exactly (Section 4), only `M_D` actually
  changes that weight (`M_D t_A = c_D t_A`). So:

  ```
  t = t_A          if D is applied before A (D precedes A in time)
  t = c_D * t_A    if D is applied after A  (D follows A in time)
  ```

  P's position is irrelevant to this outcome — consistent with P
  commuting with both D and A individually.

This produces exactly **two equivalence classes**, determined solely by
the relative time-order of D and A (P's position never matters):

```
Class 1 ("D before A"):  t = t_A = (0, 0, p)
    A o P o D   (apply D, then P, then A)
    A o D o P   (apply P, then D, then A)
    P o A o D   (apply D, then A, then P)

Class 2 ("D after A"):   t = c_D * t_A = ((1 - 4p/3) * p) in z
    P o D o A   (apply A, then D, then P)
    D o A o P   (apply P, then A, then D)
    D o P o A   (apply A, then P, then D)
```

All six orderings share the identical linear map `M`; the two classes
are distinguished *only* by the z-component of the translation, and that
distinction exists *only* because D and A do not commute (Section 4).
Within a class, orderings are identical as quantum channels (not just as
BB84 observables) for every `p`. Between classes, they differ for every
`p > 0`, by exactly the D/A commutator translation derived in Section 4.

**NUMERICAL RESULT.** `scripts/audit_full_channel_orderings.py` confirms,
at `p = 0.13` over 7 representative one-qubit states, that all
within-class Frobenius differences are `~1e-16` (floating-point zero)
and all between-class differences are `~1.593e-2`, matching the
closed-form prediction exactly.

## 6. BB84 average QBER across orderings — EXACT RESULT

The repository's BB84 states have Bloch vectors `|0>=(0,0,1)`,
`|1>=(0,0,-1)`, `|+>=(1,0,0)`, `|->=(-1,0,0)` (`bb84_states()`), each
measured in its own preparation basis (`bb84_state_qber`). For an affine
output map with linear part `M = diag(M_x, M_x, M_z)` (as derived above,
`M_x` and `M_y` are equal for all three channels here) and translation
`t = (0, 0, t_z)`:

- For `|0>`: output `z' = M_z + t_z`; QBER `= (1 - z')/2 = (1 - M_z - t_z)/2`.
- For `|1>`: output `z' = -M_z + t_z`; QBER `= (1 + z')/2 = (1 - M_z + t_z)/2`.
  Summing: `QBER(|0>) + QBER(|1>) = 1 - M_z`. **The `t_z` terms cancel
  exactly** in this sum, regardless of its value.
- For `|+>`, `|->`: the relevant Bloch component is x, and `t_x = 0` for
  every channel considered here (D, P, A all have zero x/y-translation),
  so `QBER(|+>) + QBER(|->) = 1 - M_x` **with no translation dependence
  at all**.

Therefore

```
average BB84 QBER = ( (1 - M_z) + (1 - M_x) ) / 4 = (2 - M_x - M_z) / 4
```

depends only on the (ordering-independent) linear part `M`, **not** on
the translation `t`. Since Section 5 showed `M` is identical across all
six triple orderings, the average BB84 QBER is:

```
IDENTICAL across all six orderings and both equivalence classes, for every p.
```

**This is a genuine cancellation, not an absence of a channel-level
ordering effect.** The two equivalence classes are distinct quantum
channels (Section 5) — they would be distinguishable by a measurement
sensitive to the z-translation alone (e.g. a state initially at the
Bloch-sphere center, or any state/measurement pair that does not average
`|0>`/`|1>` symmetrically) — but the specific symmetric average used by
the BB84 QBER metric is blind to exactly the quantity (translation) that
distinguishes the classes.

**NUMERICAL RESULT.** At `p = 0.13`, all six orderings give average BB84
QBER `= 0.177553283015...` to 12 decimal places (see
`results/ordering_audit/full_channel_ordering_qber.csv`).

## 7. Numerical validation methodology

`scripts/audit_full_channel_orderings.py`:

- imports `depolarizing_kraus`, `dephasing_kraus`, `amplitude_damping_kraus`,
  `apply_channel`, `bb84_states`, `projector`, `bb84_state_qber` directly
  from `qkd_noise.channels` — it does not redefine or duplicate any
  channel formula;
- enumerates all six D/P/A time-orderings and constructs each composite
  channel by direct sequential Kraus application (`apply_channel`), and,
  where `qiskit.quantum_info` is importable, also as a Qiskit `SuperOp`
  (skipped and explicitly reported, never fabricated, if qiskit is
  unavailable);
- evaluates all six orderings on the same 7-state test set used by
  `scripts/audit_channel_commutation_dp.py` (`|0><0|, |1><1|, |+><+|,
  |-><-|`, plus three general Bloch-vector states), over the same
  `p` sweep `0.01..0.10` plus `p=0.13` for a stronger separation check;
- reports pairwise Frobenius differences (density-matrix and, when
  available, SuperOp) between all `C(6,2)=15` ordering pairs, and
  classifies orderings into numerical equivalence classes at
  `TOL = 1e-12`;
- computes and reports average BB84 QBER per ordering;
- writes `results/ordering_audit/full_channel_ordering_superop.csv` and
  `results/ordering_audit/full_channel_ordering_qber.csv`, and does not
  touch the existing `dp_channel_commutation*.csv` files.

## 8. Numerical results summary

At `p = 0.13` (see Section 4/5/6 above for the exact predictions each
number is checked against):

| Comparison | Max Frobenius diff | TOL=1e-12? |
|---|---|---|
| D o P vs P o D | ~1e-16 | commute |
| P o A vs A o P | ~1e-16 | commute |
| D o A vs A o D | ~1.593e-2 | **do not commute** |
| within Class 1 (3 orderings) | ~1e-16 | equivalent |
| within Class 2 (3 orderings) | ~1e-16 | equivalent |
| Class 1 vs Class 2 | ~1.593e-2 | **distinct** |
| average BB84 QBER, all 6 orderings | equal to ~1e-12 | **identical** |

Full per-`p`, per-state values are in the generated CSVs; the script
also reports the exact numbers at runtime.

## 9. Implications for the paper

- The existing canonical result (`bb84_average_qber(p, "triple")`,
  i.e. the ordering "A o P o D") is a **valid representative of all six
  triple orderings** for the purpose of average BB84 QBER: no other
  ordering would have produced a different number for that specific
  observable. No qualification of the existing BB84 QBER analytical
  result is needed.
- However, the *composite channel itself* is not ordering-independent:
  D and A do not commute, and the six triple orderings fall into exactly
  two distinct equivalence classes as CPTP maps. If the paper (or a
  future analysis) considers any observable other than the specific
  BB84-averaged QBER — e.g. per-basis QBER asymmetry, a different
  input-state ensemble, entanglement-based protocols (E91) where the
  translation term is not automatically averaged away, or any use of the
  channel's non-unital fixed point — channel ordering could matter, and
  this should be checked case by case rather than assumed away.
- The mechanism behind D/P and P/A commuting is structural (scalar vs.
  diagonal matrices, and dephasing's exact invariance of z) rather than
  coincidental, so it is expected to be robust to small changes in the
  noise parameters, but it is specific to *this* choice of channels and
  Kraus conventions (single shared parameter `p = gamma`, no cross-qubit
  correlations) — see Open Questions.

## 10. Claims that are NOT justified by this analysis

- This analysis does **not** establish anything about multi-qubit or
  entangled-state behavior (e.g. E91's CHSH statistic) under channel
  reordering; Section 6's cancellation is specific to the single-qubit
  BB84 QBER average over exactly these four states.
- This analysis does **not** establish that D and A "almost commute" in
  any operator-norm sense beyond the specific translation term computed
  here; the stated discrepancy is exact for this parameterization
  (`gamma = p`) and would change under a different relationship between
  the depolarizing and amplitude-damping parameters.
- This analysis does **not** imply that ordering is irrelevant to secret
  key rate, PRI, or any other downstream metric in the paper; only
  average BB84 QBER was checked. Those metrics are currently computed
  from the QBER value (`metrics.py`), which is unaffected, but that is a
  consequence, not an independently re-derived fact about those metrics.

## 11. Addendum: Bit-Conditioned QBER Asymmetry (Delta Q_Z)

This addendum investigates the natural next candidate identified in
Section 10: an observable that does *not* symmetrize `|0>` and `|1>`
together, since Section 6 showed that symmetrization is exactly the step
that hides the translation term. It follows the same EXACT / NUMERICAL /
INTERPRETATION / OPEN QUESTION labeling as the rest of this document, and
does not modify any earlier section's conclusions.

### 11.1 Definition and exact derivation

Define, for a given triple ordering and noise strength `p`,

```
Delta_Q_Z(p) := QBER(|0>) - QBER(|1>)
```

using the same per-state QBER definition already used throughout this
document and already implemented as `bb84_state_qber` in
`qkd_noise.channels` (no new definition is introduced).

**EXACT RESULT.** For any affine single-qubit channel of the form
`r' = (M_x x, M_x y, M_z z + t_z)` (the form established in Sections 3
and 5 for every D/P/A composite considered here):

```
QBER(|0>) = (1 - z')/2 |_{z=1}  = (1 - M_z - t_z)/2
QBER(|1>) = (1 + z')/2 |_{z=-1} = (1 - M_z + t_z)/2

Delta_Q_Z = QBER(|0>) - QBER(|1>) = -t_z
```

This was re-derived independently by symbolic computer algebra (exact
rational/radical arithmetic, not floating point) directly from the
`depolarizing_kraus` / `dephasing_kraus` / `amplitude_damping_kraus`
Kraus operators, confirming the closed forms in Section 3 and this
formula to be exact identities, not approximations valid only near a
sampled `p`.

Combined with Section 5's two classes (`t_z = p` vs `t_z = p(1-4p/3)`,
`gamma = p`):

```
Class 1: Delta_Q_Z = -p
Class 2: Delta_Q_Z = -p(1 - 4p/3)

Delta_Q_Z(Class 1) - Delta_Q_Z(Class 2) = -p + p(1-4p/3) = -4p^2/3
```

reproducing, with the opposite sign convention (`Class1 - Class2` here
vs. `t_z` differences elsewhere), the same `4p^2/3` constant that
appears everywhere else in this document. This is not a new independent
number; it is the same D/A commutator translation from Section 4,
viewed through a different observable.

**NUMERICAL RESULT.** `scripts/audit_bit_conditioned_qber.py` confirms
`Delta_Q_Z`, the class separation, and the continued ordering-independence
of the standard average BB84 QBER against these closed forms over
`p in {0.01, ..., 0.10, 0.13}`, all to `~1e-16` (floating-point zero at
`TOL = 1e-12`).

### 11.2 Generalization to independent parameters (d, q, gamma)

Section 10 flagged that the `gamma = p` identification might make the
result an artifact of that specific choice. This was checked directly:
re-deriving Sections 3-5 with three independent parameters
`d` (depolarizing), `q` (dephasing), `gamma` (amplitude damping):

```
M_D(d) = (1 - 4d/3) I         t_D = 0
M_P(q) = diag(1-2q, 1-2q, 1)  t_P = 0
M_A(gamma) = diag(sqrt(1-gamma), sqrt(1-gamma), 1-gamma)   t_A = (0,0,gamma)
```

**EXACT RESULT.**

- `D(d)` and `P(q)` still commute exactly, for *any* `d, q` -- the proof
  in Section 4 only used that `M_D` is a scalar matrix, which holds for
  any `d`.
- `P(q)` and `A(gamma)` still commute exactly, for *any* `q, gamma` --
  the proof only used that `M_P`'s z,z entry is exactly 1 (true for any
  `q`) and that `t_A` lies purely along z (true for any `gamma`).
- `D(d)` and `A(gamma)` still do not commute, and the linear part `M` is
  still identical regardless of composition order (diagonal/scalar
  matrices commute regardless of their entries). The translation
  difference generalizes cleanly to a **bilinear cross term**:

```
t_{D o A} - t_{A o D} = (0, 0, -4 d*gamma/3)
```

  i.e. `p^2` is replaced by the product `d*gamma`, with **no dependence
  on `q` whatsoever** -- consistent with dephasing being "invisible" to
  this effect regardless of its own strength.

- The two triple-ordering classes persist exactly, with:

```
Class 1 (D before A):  t_z = gamma
Class 2 (D after A):   t_z = gamma * (1 - 4d/3)
Delta_Q_Z class separation = -4*d*gamma/3
```

**This is the important finding of this sub-task: the effect does NOT
depend on the artificial identification `d = gamma = p`.** It survives
fully general, independent depolarizing and amplitude-damping strengths,
as a clean exact bilinear term `d*gamma`. The `p^2` appearing throughout
the rest of this document is simply this bilinear term evaluated at
`d = gamma = p`, not a special coincidence of that choice. The dephasing
parameter `q` plays no role in the magnitude of the effect at any point
-- only in confirming (via Section 4) that dephasing does not interfere
with it.

### 11.3 General (unequal) BB84 state-preparation weighting

Section 6 showed the *uniform* four-state average cancels `t_z`. Section
10 (Task 9) asked whether a general weighting exposes it, and Task 5 of
the current request asks for the exact condition. For weights
`w_0, w_1, w_+, w_-` with `w_0+w_1+w_++w_- = 1`:

**EXACT RESULT.**

```
weighted average QBER
  = (1/2)[ (w_0+w_1)(1-M_z) + (w_++w_-)(1-M_x) ] + (t_z/2)(w_1 - w_0)
```

so the coefficient of `t_z` in the weighted average is exactly
`(w_1 - w_0)/2`. Therefore:

```
t_z cancels  <=>  w_0 = w_1        (regardless of w_+, w_-, and regardless
                                     of how the total Z-basis vs X-basis
                                     weight (w_0+w_1) vs (w_++w_-) is split)
```

**INTERPRETATION.** This sharpens (and partially corrects the naive
expectation behind) Section 10's suggestion: it is specifically an
imbalance between `w_0` and `w_1` -- the probability of Alice sending bit
0 vs bit 1 within the Z basis -- that exposes the translation, with
sensitivity exactly `(w_1-w_0)/2 * t_z`. A *basis-biased* BB84 variant
(e.g. unequal Z-basis vs X-basis sampling probabilities, as used in
efficient BB84) does **not** by itself expose the effect if `w_0 = w_1`
still holds within the Z basis; only an imbalance in the *bit value*
distribution does. Uniform BB84 (`w_0=w_1=w_+=w_-=1/4`) is simply the
special case `w_1-w_0=0`, confirming Section 6 as one specific point on
this family rather than an isolated cancellation.

### 11.4 Operational meaning: is Delta_Q_Z a genuine QKD observable?

This sub-section answers Task 3 of the current request directly and
conservatively; no external literature is cited, and no claim is made
that has not been checked against the assumptions already encoded in
this repository.

1. **Is `QBER(|0>)` vs `QBER(|1>)` a physically meaningful bit-conditioned
   statistic?** Yes, in the narrow sense that it is a well-defined
   function of the sifted-key data: Alice knows the bit value she sent
   for each sifted round, and Bob's measurement outcome after basis
   reconciliation lets both parties compute an error rate conditioned on
   that bit value. It requires no assumption beyond what standard BB84
   post-processing (sifting) already provides.
2. **Is it normally observable in BB84 implementations?** In principle,
   yes -- nothing about the protocol prevents computing it from existing
   sifted-key statistics. In practice, standard treatments (including
   `bb84_average_qber` and `metrics.py` in this repository) report only
   the *symmetric* QBER, so this quantity is not currently computed or
   exposed anywhere in this project's existing pipeline; it would need
   to be added as new instrumentation, not merely re-read from existing
   outputs.
3. **Does it survive classical post-processing, or is it normally
   symmetrized away?** Under the assumptions used in this project
   (`bb84_average_qber` uses `np.mean` over all four states uniformly),
   it is symmetrized away by construction before it reaches any
   downstream metric. It is not that the physical asymmetry disappears
   during error correction/privacy amplification in some deep sense --
   it is that the *summary statistic currently computed* in this
   repository never separates `|0>` from `|1>` in the first place.
4. **Is it related to detector asymmetry, source bias, basis-dependent
   effects, or finite-key parameter estimation?** Functionally, a
   bit-value-dependent error rate is the same *category* of quantity
   that appears in discussions of detector-efficiency mismatch and
   similar side channels in the wider QKD security literature -- but
   this document makes no claim about that literature beyond noting the
   structural analogy, and does not cite specific results, since no
   literature search was performed and no such claim can be verified
   here. In this repository's model, `Delta_Q_Z` arises purely from
   channel-ordering-induced non-unital bias (Section 4), which is a
   different physical mechanism from detector hardware asymmetry, even
   if the resulting statistic has the same mathematical shape.
5. **Are we allowed to call it a "QKD observable" under the assumptions
   already used in this paper?** With qualification: it is directly
   expressible using functions already present in
   `qkd_noise.channels` (`bb84_state_qber` already computes
   `QBER(|0>)` and `QBER(|1>)` individually, before `bb84_average_qber`
   averages them), so it requires **no new physical assumption** to
   define or compute. However, calling it an established *security
   parameter* would overclaim its current role: nothing in this
   repository's existing protocol model (or, to this document's
   knowledge without a literature search, in standard BB84 security
   proofs) uses a bit-conditioned QBER decomposition as an input to a
   security bound. **The accurate characterization is: a mathematically
   well-defined and directly computable diagnostic statistic, derived
   from the existing per-state QBER machinery, that is not currently
   part of this project's (or, as far as this analysis can determine,
   the standard) security-parameter set.**

### 11.5 Verdict

Per the request to give a clear verdict rather than proceeding
unprompted to further research:

**Verdict: (B) Mathematically valid, exact, and more general than the
original `p=gamma` special case -- but currently a supporting/diagnostic
result, not a standalone theorem.**

Reasoning:

- The mathematics is fully rigorous and exact (Sections 11.1-11.3,
  cross-checked symbolically and numerically), and Section 11.2 shows it
  is *not* an artifact of setting `d=gamma=p` -- it is a genuine bilinear
  `d*gamma` effect.
- However, Section 11.4 shows `Delta_Q_Z` is not, at present, tied to
  any security bound, key-rate formula, or other operationally-consequential
  quantity in this project or (to the extent checkable here) in standard
  BB84 treatments. It is a real, exact property of the channel/observable
  pair, but its significance depends entirely on whether a downstream use
  (a modified security analysis, a side-channel argument, a finite-key
  estimator that does track bit-conditioned statistics) is developed for
  it. That step has not been done.
- It is **not** "mostly a diagnostic artifact" in the dismissive sense of
  verdict (C): the underlying channel-level classes are real and the
  `Delta_Q_Z` formula is exact and non-trivial (Section 11.1-11.2 show it
  survives full generalization). But absent a concrete link to a
  security or key-rate consequence, it does not yet rise to (A) "strong
  candidate for a new theorem" on its own -- it is best framed as a
  precise structural lemma supporting a future result, not a headline
  result by itself.
- Recommendation for next steps (not executed in this task, per the
  request to stop here): if this is to become a stronger result, the
  most direct path is connecting `Delta_Q_Z` (or the underlying `t_z`
  asymmetry) to a quantity that *does* feed a security bound -- e.g.
  whether a finite-key estimator that (mis)assumes bit-symmetric noise
  incurs a bias of order `d*gamma`, rather than further generalizing the
  channel algebra, which is now fairly complete.

## Open Questions

- Section 11.2 resolves the `d = gamma = p` generalization question
  raised below (RESOLVED: the effect survives as a clean `d*gamma`
  bilinear term). The original bullet is retained for the historical
  record.
  - ~~Whether the "D before/after A" classification generalizes to
    composite channels with *independent* depolarizing/amplitude-damping
    parameters (`p != gamma`) is not addressed here.~~
- Whether any single-qubit observable other than the BB84-symmetric QBER
  average would detect the translation-level difference between the two
  classes is now partially addressed (Section 11.1: `Delta_Q_Z`; Section
  11.3: general weighted averages). Still open: whether `Delta_Q_Z` (or
  the general weighted-average formula of Section 11.3) has any
  consequence for a security bound, finite-key estimator, or key rate --
  this document does not attempt that connection (see Section 11.5).
- E91/CHSH ordering-dependence remains entirely unexamined, per the
  explicit scope limitation of this task.
