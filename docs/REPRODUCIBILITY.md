# Reproducibility

SymmClimateNet separates development/model selection from post-lock characterization.

The package does **not** redistribute the underlying primary or external weather observations. Users must obtain data under the applicable provider terms and construct tensors with matching station/feature semantics.

For software verification:

```bash
python -m pip install -e ".[dev]"
pytest -q
symmclimatenet info
```

The PyPI distribution and GitHub/Zenodo release should carry the same version tag (`v1.0.0`).
