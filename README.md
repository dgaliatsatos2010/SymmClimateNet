# SymmClimateNet

**SymmClimateNet (Symmetry-Guided Spatio-Temporal Climate Network)** is a PyTorch research model for multi-station daily climate forecasting. Its public v1.0.0 architecture was selected by a predeclared model-lock protocol and was not changed after frozen benchmarking or external-source observational validation.

## Core idea

SymmClimateNet treats station ordering as nuisance structure and uses a shared causal station encoder with permutation-invariant aggregation. Spatial information enters through a reciprocal operator constrained to be symmetric by construction, followed by a persistence-residual forecast head and predictive quantiles.

The public release intentionally exposes only the **locked symmetric-spatial architecture**. The earlier antisymmetric branch was a development ablation and is not part of v1.0.0.

## Installation

From a local checkout:

```bash
pip install .
```

After the PyPI release:

```bash
pip install symmclimatenet
```

## Quick start

```python
import numpy as np
import torch
from symmclimatenet import build_model

coords = np.array([
    [37.98, 23.72], [40.64, 22.94], [39.64, 22.42], [38.25, 21.73],
    [39.67, 20.85], [40.85, 25.87], [40.47, 25.52], [40.90, 24.43],
], dtype=np.float32)

model = build_model(n_features=7, station_coordinates=coords, target_feature_index=0)
x = torch.zeros(2, 30, 8, 7)
out = model(x, target_station_index=0)
print(out["quantiles"])
```

Inspect the exact public architecture lock:

```bash
symmclimatenet info
```

## Validation status

The archived evidence documents three distinct roles:

- **Stage B:** development/model-selection evidence used to lock the architecture. The symmetric-only model was selected by a predeclared tie-breaking rule favoring parsimony.
- **Stage C:** frozen characterization against eight baselines. The locked model ranked 1st, 2nd and 3rd at 1-, 3- and 7-day horizons, respectively.
- **Stage D:** external-source observational validation. The locked model ranked 1st at 1 day (MAE 0.9714 °C), was essentially tied for best at 3 days (1.8350 °C), and ranked 3rd at 7 days (2.1848 °C).

Because the Stage-D dates had been seen previously from the primary data source, Stage D is **external-source confirmation rather than untouched temporal confirmation**. No universal state-of-the-art claim is made.

See `docs/VALIDATION.md` and the derived files under `evidence/`.

## Reproducibility boundary

Raw weather observations and pretrained study checkpoints are not redistributed in the PyPI package. This repository preserves the tested reference implementation, architecture lock and derived evidence summaries. Data must be obtained under the original providers' terms.

## Software citation

See `CITATION.cff`. A version-specific Zenodo DOI will be added to the citation metadata after the GitHub `v1.0.0` release is archived.

## License

MIT License for the software. Data retain their original licenses and attribution requirements.
