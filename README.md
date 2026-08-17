# Composite-Noise QKD — Phase 2

Implementation baseline for reproducing the uploaded paper, *Composite Noise Interaction in Discrete-Variable QKD*.

## Phase 2 scope

1. Kraus noise channels
2. Sequential channel composition
3. BB84 state-level QBER reference calculations
4. Practical-parameter calculations
5. Secret-key-rate and PRI metrics
6. Automated consistency tests

Protocol-level BB84/B92/E91 circuit simulation comes after the noise engine is validated.

## Reproduction rule

The code does **not silently repair contradictions in the paper**.

Known unresolved items:
- B92 conclusive-outcome mapping is not fully specified.
- E91 CHSH setting assignment is not fully specified.
- Reported PRI values are inconsistent with Eq. (20).
- The stated super-additivity theorem is not reproduced by the stated Kraus maps.
- The dark-count attenuation expression appears inconsistent with the stated dB/km parameterization.

## Environment

The paper reports Qiskit Aer 0.13. The legacy-Qiskit environment is kept separate from the pure NumPy reference engine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/validate_noise.py
```
