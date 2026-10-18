# SymmClimateNet architecture

SymmClimateNet v1.0.0 is the publication-locked symmetric-spatial model selected before the frozen benchmark and external-source validation.

The network uses a shared causal dilated temporal encoder for every station, permutation-invariant mean/max aggregation, a persistence-residual forecast head, and a reciprocal spatial interaction operator that is symmetric by construction. For head $h$, the learned operator satisfies $S^{(h)}_{ij}=S^{(h)}_{ji}$. Station ordering is treated as nuisance structure: consistently permuting station histories, coordinates and the target-station index must not change the scalar target forecast.

The earlier antisymmetric development branch is not part of public v1.0.0 because the predeclared Stage-B lock rule resolved the near-tie in favor of the simpler symmetric-only architecture.

## Locked defaults

- history: 30 days
- hidden dimension: 48
- prediction-head hidden dimension: 64
- spatial message dimension: 12
- symmetric heads: 4
- auxiliary hidden dimension: 32
- predictive quantiles: 0.05, 0.50, 0.95
- mode: `set_temporal_plus_symmetric_spatial`

The exact lock metadata are exposed by `symmclimatenet.locked_spec_dict()`.
