# Stage E — zero-shot Italian cross-network transfer: completed results

## Integrity

Stage E completed the full predeclared matrix without retraining or fine-tuning on Italian data:

- external network: eight predeclared Italian city anchors (Turin, Milan, Bologna, Florence, Rome, Naples, Bari, Palermo)
- horizons: 1, 3, and 7 days
- seeds: 13, 47, and 97
- frozen neural variants: raw control, symmetric spatial, symmetric+antisymmetric spatial, unconstrained spatial
- neural cells: 288/288
- persistence cells: 24/24
- all neural cells were zero-shot; no neural checkpoint was retrained on Italian observations
- Stage-E external tensor SHA-256: `c623f7e1ff1c48bcb63f577350317f5a71c37134026c42b8a34a3fa6c435f36b`

The locked symmetric model retained exact structural behavior: maximum permutation-invariance error was (2.38\times10^{-7}), maximum symmetry error was 0.0, and minimum symmetric-operator RMS was 0.7632.

## Horizon-level mean MAE across eight targets

| Horizon | Raw control | Locked symmetric | Symmetric+antisymmetric | Unconstrained spatial | Persistence |
|---|---:|---:|---:|---:|---:|
| 1 day | **1.2300** | 1.2568 | 1.2994 | 1.2650 | 1.2605 |
| 3 days | 2.0646 | **2.0306** | 2.0663 | 2.0475 | 2.1245 |
| 7 days | **2.4753** | 2.4773 | 2.4992 | 2.5134 | 2.5926 |

Across all 24 target×horizon summaries, the locked symmetric model had the lowest unweighted mean MAE (1.9216 °C), narrowly ahead of the raw control (1.9233 °C), followed by the unconstrained spatial model (1.9420 °C), symmetric+antisymmetric model (1.9550 °C), and persistence (1.9925 °C). This pooled value is descriptive because horizons differ in scale.

Among the four frozen neural variants, the locked symmetric model was best in 9/24 target×horizon cells (37.5%). The raw control was best in 13/24 and the unconstrained model in 2/24. The horizon pattern was strongly heterogeneous: raw control led all eight targets at 1 day, whereas the locked symmetric model led 6/8 targets at 3 days and 3/8 at 7 days.

## Primary moving-block bootstrap

The primary date-level comparison averaged loss across seeds within each target and then across available targets, requiring at least six of eight targets per date. A 14-day moving-block bootstrap with 2,000 replicates was used; Holm correction was applied across four comparators within each horizon. Negative values favor the locked symmetric model.

| Horizon | Comparator | Delta MAE (°C) | 95% CI | Holm-adjusted p |
|---|---|---:|---|---:|
| 1 | Persistence | -0.0068 | [-0.0358, 0.0179] | 0.483 |
| 1 | Raw control | +0.0268 | [0.0194, 0.0333] | <0.001 |
| 1 | Symmetric+antisymmetric | -0.0425 | [-0.0477, -0.0370] | <0.001 |
| 1 | Unconstrained spatial | -0.0083 | [-0.0118, -0.0044] | <0.001 |
| 3 | Persistence | -0.0999 | [-0.1657, -0.0383] | 0.003 |
| 3 | Raw control | -0.0343 | [-0.0542, -0.0134] | 0.003 |
| 3 | Symmetric+antisymmetric | -0.0357 | [-0.0466, -0.0230] | <0.001 |
| 3 | Unconstrained spatial | -0.0164 | [-0.0315, -0.0020] | 0.020 |
| 7 | Persistence | -0.1251 | [-0.2366, -0.0090] | 0.090 |
| 7 | Raw control | +0.0017 | [-0.0133, 0.0173] | 0.806 |
| 7 | Symmetric+antisymmetric | -0.0226 | [-0.0436, 0.0001] | 0.104 |
| 7 | Unconstrained spatial | -0.0367 | [-0.0580, -0.0161] | <0.001 |

## Interpretation

Stage E supports a narrower and stronger transportability claim than a simple accuracy win. The locked symmetric spatial branch is **not uniformly superior to the raw temporal backbone**: the raw control is significantly better at one day and essentially tied at seven days. However, the symmetric constraint transfers more robustly than the information-matched unconstrained spatial branch at all three horizons, with Holm-adjusted confidence intervals excluding zero each time. At three days, the locked symmetric model is significantly better than every predeclared comparator.

This pattern supports the interpretation that reciprocal symmetry can regularize cross-station information sharing under a new station graph, especially at the medium forecast horizon, while also showing that spatial augmentation itself is not universally beneficial.

## Uncertainty and limitations

Mean locked-model 90% interval coverage in Stage E was 78.1% at 1 day, 86.6% at 3 days, and 85.3% at 7 days, confirming that zero-shot point-forecast transfer is stronger than uncertainty-calibration transfer.

Several selected Meteostat stations lack complete precipitation and/or wind-gust channels. These missing predictors were not imputed post hoc; the frozen pipeline handled them exactly as predeclared. This makes Stage E a stringent geographic/network and feature-availability stress test, but it also limits attribution of performance differences solely to geography.

Stage E is an untouched geographic/network transfer test with respect to the Italian stations and observations at model lock. It is **not** an untouched temporal-period test because the calendar years overlap periods used elsewhere in the project.
