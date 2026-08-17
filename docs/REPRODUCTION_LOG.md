
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
