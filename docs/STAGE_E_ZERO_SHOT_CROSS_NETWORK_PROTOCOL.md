# Stage E — predeclared zero-shot cross-network geographic transfer protocol

## Purpose

Stage E tests whether the already locked SymmClimateNet architecture transports to an entirely different station network in another country **without retraining, fine-tuning, recalibration, feature redefinition, or architecture changes**. This is a post-lock stress test of geographic/network transportability and of the symmetry-based inductive bias; it is not a new model-selection stage.

## Frozen scientific question

Can the Greek-trained, publication-locked symmetric-spatial model transfer zero-shot to an independent Italian station network, and does the symmetric spatial inductive bias transfer at least as well as the previously trained raw, symmetric+antisymmetric, and unconstrained Stage-B controls?

No Stage-E result may alter the locked model.

## Independent geographic network

The network is fixed before model evaluation using eight Italian city anchors:

1. Turin — 45.0703, 7.6869
2. Milan — 45.4642, 9.1900
3. Bologna — 44.4949, 11.3426
4. Florence — 43.7696, 11.2558
5. Rome — 41.9028, 12.4964
6. Naples — 40.8518, 14.2681
7. Bari — 41.1171, 16.8719
8. Palermo — 38.1157, 13.3615

These locations were not part of the Greek CLIMPACT development network or any earlier SymmClimateNet model-selection stage.

## External observations and station selection

Source: Meteostat 2.1.5 station observations.

- Model-data substitution: disabled.
- Spatial interpolation: not used.
- Support window: 2022-11-20 onward, solely to supply the 30-day history before the first 2023 target.
- Frozen test period: 2023-01-01 through 2025-12-31.
- For each anchor, candidate stations are considered in ascending geographic distance.
- Search radius: 100 km.
- Candidate limit: 20.
- External station IDs must be unique across the eight anchors.
- The first unique station with at least 90% mean-temperature completeness over the frozen test period is selected.
- Predictive performance is never used for station selection.
- Non-target feature missingness is retained as observed and handled by the frozen model pipeline.

The exact selected station IDs, coordinates, distances, feature completeness, candidate log, and source-tensor hash are archived.

## Feature mapping

The same eight canonical variables and units are used:

- daily mean 2-m temperature (°C)
- daily maximum temperature (°C)
- daily minimum temperature (°C)
- daily mean relative humidity (%)
- daily mean sea-level pressure (hPa)
- daily precipitation (mm/day)
- daily mean wind speed (km/h)
- peak wind gust (km/h)

Primary-source feature means and standard deviations fitted through 2020-12-31 are reused unchanged. No Italian observation is used to refit normalization.

## Frozen models

Stage E performs inference only. Exact dev5 Stage-B checkpoints are loaded for each horizon and seed:

- set_temporal_raw_control
- set_temporal_plus_symmetric_spatial — the publication-locked proposed model
- set_temporal_plus_sym_asym_spatial
- set_temporal_plus_unconstrained_spatial

Persistence is included as a no-training reference.

None of the four neural checkpoints is retrained or fine-tuned on Italian data. Stage-E labels are not used for early stopping, calibration, or parameter selection.

## Evaluation matrix

Targets: all eight Italian stations.

Horizons: 1, 3, and 7 days.

Seeds: 13, 47, and 97 for the four frozen neural checkpoints.

Primary matrix: 8 targets × 3 horizons × 3 seeds × 4 frozen neural variants = 288 neural cells, plus persistence for every target/horizon.

## Outcomes

Per target/horizon/model:

- MAE
- RMSE
- R²
- bias
- nominal 90% interval coverage and interval width for neural models
- runtime
- checkpoint SHA-256
- structural diagnostics

### Primary comparison

For each horizon and comparator, absolute errors are first averaged across the three seeds and then across available target stations for each test date. The primary effect is the date-level loss difference:

symmetric locked MAE − comparator MAE

Comparators:

1. persistence
2. set-temporal raw control
3. symmetric+antisymmetric control
4. unconstrained spatial control

A moving-block bootstrap with block length 14 days and 2,000 replicates estimates the mean difference and 95% confidence interval. Holm correction is applied across the four comparator tests within each horizon.

### Secondary descriptive analyses

- target-specific MAE and rank by horizon
- fraction of target×horizon cells in which the locked symmetric model is best among the four frozen neural variants
- relative improvement versus persistence
- cross-network interval coverage
- permutation-invariance and operator-symmetry diagnostics

## Interpretation boundary

Stage E is an **untouched geographic/network transfer test** with respect to the Italian stations and their observations at the time of model lock. The calendar years overlap periods used elsewhere in the broader project, so Stage E does not establish untouched temporal generalization.

A favorable Stage-E result supports geographic/network transportability and tests whether the symmetric spatial inductive bias transfers under a new station graph. An unfavorable result is reported as a limitation and may not be used to modify SymmClimateNet v1.0.1.

## Non-negotiable firewall

- Architecture and hyperparameters remain frozen.
- No Italian label may trigger model or preprocessing changes.
- Station selection cannot use predictive performance.
- No threshold in this protocol may be altered after Stage-E results are inspected.
- Stage E cannot reopen the Stage-B model-lock decision.
