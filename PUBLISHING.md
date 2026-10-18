# Publishing checklist for v1.0.0

1. Verify the GitHub test workflow passes.
2. In PyPI, configure a Trusted Publisher for repository `dgaliatsatos2010/SymmClimateNet`, workflow `.github/workflows/publish-pypi.yml`, environment `pypi`.
3. Create GitHub tag/release `v1.0.0`. The release event builds and publishes `symmclimatenet==1.0.0` to PyPI.
4. Connect the GitHub repository to Zenodo and archive the `v1.0.0` release.
5. Add the resulting version DOI and concept DOI to `CITATION.cff` and README.
6. Record Git commit SHA, PyPI release URL, Zenodo DOI, and SHA-256 manifest in the manuscript reproducibility section.
