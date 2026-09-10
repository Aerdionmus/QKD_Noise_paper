# Composite-Noise QKD — Phase 2

Repository for the validated composite-noise QKD analysis accompanying
*Composite Noise Interaction in Discrete-Variable QKD*.

## Frozen scientific scope

The current scientific focus is:

1. exact affine analysis of depolarizing (`D`), dephasing (`P`), and
   amplitude-damping (`A`) channel composition;
2. exact classification of all six `D/P/A` orderings into two
   channel-equivalence classes;
3. the ordering defect
   `Delta t_z = 4 d gamma / 3` (and `4 p^2 / 3` when `d=q=gamma=p`);
4. validated operational consequences for conditioned BB84 statistics,
   two-sided Bell correlations, concurrence, fixed-setting CHSH, and
   channel discrimination; and
5. Werner-state visibility robustness.

The repository also retains legacy BB84/B92/E91 protocol simulations and
paper-equation audits as separately identified reproduction material. No new
models, protocols, metrics, or experiments are implied by this frozen scope.

### Exact ordering result

`D` and `P` commute, `P` and `A` commute, and `D` and `A` do not commute.
The six compositions form exactly two classes:

- Class 1: `A o P o D`, `A o D o P`, `P o A o D`
- Class 2: `P o D o A`, `D o A o P`, `D o P o A`

All six orderings have the same linear Bloch map; the classes differ only in
the affine `z` translation. The symmetric average BB84 QBER therefore agrees
across orderings, while the Z-basis conditioned QBER and the two-sided
correlation observables can distinguish the classes.

## Reproduction rule

The code does **not silently repair contradictions in the paper**.

Known unresolved items:
- B92 conclusive-outcome mapping is not fully specified.
- The legacy E91 protocol implementation and its CHSH-setting convention
  require later reconciliation with the separate fixed-setting ordering study.
- Reported PRI values are inconsistent with Eq. (20).
- The stated super-additivity theorem is not reproduced by the stated Kraus maps.
- The dark-count attenuation expression appears inconsistent with the stated dB/km parameterization.
- The DIQKD expression used by the ordering study is an application of an
  established literature bound, not a newly proved security theorem.

## Environment

The paper reports Qiskit Aer 0.13. The legacy-Qiskit environment is kept separate from the pure NumPy reference engine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/validate_noise.py
```
