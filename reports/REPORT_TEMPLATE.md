# Cross-Dataset Generalization of Machine-Learning Models for EV Battery SOH Prediction

**Status: TEMPLATE — no experimental results have been collected yet.**

## Abstract
[Complete only from real benchmark results.]

## 1. Introduction and research questions
- Does within-dataset performance reliably predict cross-dataset performance?
- How much does performance degrade across LCO, NCM and LFP and operating protocols?
- How many target batteries are needed for effective adaptation?

## 2. Data sources and provenance
Record download link, source version, license, paper, cell IDs, nominal capacity, measurement cadence and excluded/duplicate cells separately for NASA/CALCE/XJTU/Oxford/MATR.

## 3. Harmonization and task definition
- SOH = 100 x measured full-discharge capacity (Ah) / rated nominal capacity (Ah).
- Common 80% SOH EOL threshold; define whether observations after crossing are included.
- Define prediction time (same-cycle health estimation vs next-cycle forecast).
- Define inputs used in all models. Do not use same-cycle capacity/target SOH as inputs.
- Describe missing-feature handling without fabrication.
- Keep original observed characterization cycle indexes; no label interpolation in strict benchmark.

## 4. Protocol
- Batteries split, never individual cycles.
- Within-dataset test (source train/validation/test groups).
- Cross-dataset 5x5 pairwise zero-shot matrix, source-only model selection.
- LODO: train on four datasets; hold out all cells of the fifth.
- Adaptation: 0, 1, 3, 5, 10 target train batteries, only if feasible, target test cells never used for adaptation.
- Model comparison: Mean, Ridge, ElasticNet, XGBoost, LSTM, CNN, Transformer.
- Five seeds [11,22,33,44,55]. Metrics MAE, RMSE, MAPE, mean +/- std over seeds.
- CORAL adaptation sample protocol and source vs target support statistics.

## 5. Results (from saved real metrics only)
### Within-dataset metrics
[Insert table from actual within metrics]
### Cross-dataset transfer matrices
[Insert measured per-model 5x5 matrices; indicate diagonal holdout method]
### Leave-one-dataset-out
[Insert measured results]
### Few-shot adaptation
[Insert observed transfer curves and infeasible K settings]
### Failure analysis
[Include chemistry, temperature and charging rate only where actually measured; confounding limits]

## 6. Discussion and limitations
- Small numbers of held-out batteries in Oxford/CALCE
- Nonstandard capacity measurement cadence, across protocol differences
- Nominal capacity uncertainty and battery chemistry confounding
- Cross-batch MATR cell identity checking
- Model failure cases and weak baseline performance

## 7. Reproducibility and references
Python version, dependencies, random seeds, source download dates and exact command lines. Cite original dataset licenses/papers.
