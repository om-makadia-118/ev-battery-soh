# Implementation verification status

- [x] Canonical dataset schema / SOH / EOL flags / capacity sanity and duplicate checks
- [x] NASA discharge/charge MATLAB parser (synthetic fixture tested)
- [x] Oxford MATLAB v5 and HDF5/MATLAB v7.3 readers (synthetic fixtures tested)
- [x] Trusted-source pickle inspection for XJTU file schema discovery
- [x] Native loader for BatteryLife-standardized XJTU pickle records (synthetic fixture tested; real inspect output matches expected schema)
- [x] MATR v7.3 `batch/summary` reader (synthetic fixture tested); cross-batch battery identity still needs verification
- [x] Per-cycle CSV / XLSX mapping loader for verified cycle-summary exports
- [x] Battery-disjoint splits and prohibition of current-cycle SOH/capacity predictor leakage
- [x] Mean, Ridge, ElasticNet and XGBoost baselines
- [x] LSTM, 1D-CNN and Transformer model code; CPU smoke training tested when PyTorch is installed
- [x] Within-dataset / cross-dataset / matrix / leave-one-dataset-out experiment runners
- [x] Target-disjoint few-shot adaptation, classical pooled/scratch, PyTorch fine-tuning and classical CORAL/normalization
- [x] Metrics, predictions, failure logs, transfer heatmaps and report plotting
- [x] CPU-only classical dependency setup separated from optional PyTorch
- [x] Google Colab notebook for GPU deep-learning runs
- [x] Synthetic smoke tests (explicitly NON-research)
- [x] Native CALCE CS2 Arbin Excel parser (real CS2_33 workbook validated; synthetic multi-file cycle-reset tests pass)
- [x] Native CALCE CS2 CADEX TXT parser (real CS2_8 file validated; CS2_21 real-file spot check pending)
- [ ] Acquisition and validation of all five full dataset selections
- [ ] Canonical MATR battery identity across batch files
- [ ] Verified common physical predictor features beyond cycle index
- [ ] Complete five-seed real benchmark matrix and LODO
- [ ] Full adaptation experiments and result-backed domain failure analysis
- [ ] Results-backed final paper
- [ ] Feature-level CORAL/normalization for sequence models (not implemented; do not claim it)

## Inventory corrections to the initial report

- The four NASA files supplied by the team are four batteries, not 28.
- The 15 `XJTU_3C_battery-*.pkl` files correspond to the 3C / Batch 2 subset, not all 55 XJTU batteries.
- The CALCE 144-cell Storage Data page is a different experiment from the report's 15-cell CS2 cycling collection.
- CALCE CS2_8 and CS2_21 are CADEX `.txt` files; the other 13 CS2 cells use Arbin Excel workbooks. The CADEX parser integrates measured current over elapsed time and does not treat the file's `Capacity` progress field as Ah.
- CALCE's general page lists 1.1 Ah for CS2. The supplied CS2_8 file integrates to about 1.10–1.12 Ah, while the first available CS2_33 workbook records about 0.891 Ah; this may reflect prior degradation or cell/source details. The report's 1.1 Ah assumption is retained by default pending source-metadata validation.

**Current stage:** XJTU pickle schema, CALCE Arbin Excel parser and CALCE CADEX parser have been checked against real user-supplied files. The CS2_21 CADEX layout still needs a spot check; MATR identity validation remains. The user's complete raw dataset collection has not yet been run through the full preparation pipeline.
