# Validation design

The public release preserves the model-lock boundary used in the study.

## Stage B — model lock

Architecture selection used five reused development folds, horizons 1/3/7 and seeds 13/47/97. The predeclared rule selected the symmetric-spatial variant. Stage-B evidence is development/model-selection evidence, not confirmatory evidence.

## Stage C — frozen development benchmark

The locked model was compared with Persistence, Ridge, Random Forest, HistGradientBoosting, MLP, LSTM, TCN and Transformer. The proposed model ranked first at horizon 1, second at horizon 3 and third at horizon 7, with mean rank 2.0 across horizons. Because 2023–2025 had been inspected in earlier development, Stage C is frozen characterization rather than untouched confirmation.

## Stage D — external-source observational validation

The architecture and hyperparameters remained locked. An independent observational source (Meteostat station observations, with model-data substitution disabled in the study workflow) was used to characterize source transportability over dates already seen from the primary source. Therefore Stage D supports external-source confirmation, not untouched temporal confirmation.

At horizon 1, the locked model had mean MAE 0.9714 °C and ranked first. At horizon 3 it was essentially tied with the best Transformer result (1.8350 vs 1.8343 °C). At horizon 7 it ranked third (2.1848 °C).

Claims should therefore emphasize short-horizon advantage and cross-source competitiveness, not universal superiority.
