# Cross-Dataset Generalization of EV Battery State-of-Health Prediction

A reproducible Python research pipeline based on the supplied BMSCE Project Work 2 report.

> **Research integrity:** no real benchmark scores are bundled or claimed. Synthetic demo data are only for software smoke tests. Dataset acquisition, source metadata, actual cell counts and feature compatibility must be verified before research claims are made.

## Hardware plan — no local GPU required

- **Windows PC (CPU):** dataset inspection, raw-data conversion, SOH calculation, quality checks, classical models (mean/Ridge/ElasticNet/XGBoost), battery-level splitting, plots and result aggregation.
- **Google Colab GPU:** LSTM, 1D-CNN and Transformer benchmark runs. The notebook is `notebooks/colab_deep_learning.ipynb`.
- **Optional local deep learning:** install CPU-only PyTorch using `requirements-cpu-torch.txt`; it will run without CUDA but may be slower.

## Local setup (Windows PowerShell)

```powershell
cd ev-battery-soh
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Optional CPU-only PyTorch (not needed for classical experiments):

```powershell
pip install -r requirements-cpu-torch.txt
```

Run tests and the clearly labelled synthetic smoke test:

```powershell
$env:PYTHONPATH = "."
pytest -q
python -m battery_soh demo
```

## Current dataset decisions

Read **[docs/DATASETS.md](docs/DATASETS.md)** before downloading more files. Important corrections to the original report's table:

- NASA files `B0005/B0006/B0007/B0018` represent **4 cells**, not 28.
- The 15 `XJTU_3C_battery-*.pkl` files are XJTU Batch 2 (3C), not the full 55-cell XJTU dataset.
- CALCE must be **CS2 cycling**, not the separate 144-cell Storage Data experiment. All 15 CS2 archives have been downloaded. Both Arbin Excel and CADEX tab-separated `.txt` parsers are implemented; the CADEX reader has been validated against a real CS2_8 file, and CS2_21 still needs a real-file spot check.
- Oxford `.mat` files are read by Python; MATLAB is not required.
- MATR is the original 124-cell Severson 2019 collection (three batch `.mat` files).

## Prepare real data

Place source files under `data/raw/<dataset>/` as described in `docs/DATASETS.md`.

```powershell
python -m battery_soh inspect --path "data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat"
python -m battery_soh inspect --path "data/raw/xjtu/XJTU_3C_battery-1.pkl"
python -m battery_soh prepare --raw data/raw
```

The XJTU loader supports the standardized BatteryLife pickle structure. Inspect one file first; if it matches the documented structure, all matching `XJTU*.pkl` files are loaded automatically. The CALCE Arbin parser reads all date-named `Channel_*` sheets per cell, corrects cycle-index resets across files, derives capacity from the cumulative discharge counter, and keeps only full-discharge capacity checks. The CADEX parser reads CS2_8/CS2_21 tab-separated logs, integrates discharge current over time to obtain Ah, and uses the `Capacity` field only as a step-completion heuristic—not as an Ah label. The CS2_8 parser has been checked against a real user-supplied file; inspect a CS2_21 file to confirm the same layout.

## CPU classical experiments

After a real `data/processed/unified.csv` exists:

```powershell
python scripts/run_classical_cpu.py --seeds 42
```

For the five-seed classical matrix:

```powershell
python scripts/run_classical_cpu.py --seeds 11 22 33 44 55
```

This runs mean, Ridge, ElasticNet and XGBoost on the 5×5 within/cross matrix. Inspect `failures.csv` before interpreting the metrics.

## Deep-learning experiments in Colab

1. Upload the extracted project folder to `MyDrive/ev-battery-soh`.
2. Ensure `data/processed/unified.csv` is present there.
3. Open `notebooks/colab_deep_learning.ipynb` in Google Colab.
4. Select **Runtime → Change runtime type → T4 GPU** if available.
5. Run the pilot first; only then run the five-seed matrix and LODO cells.

The notebook writes metrics, predictions, failures and figures back to Google Drive.

## Research protocols

- One row per observed capacity characterization, not one row per raw tester timestamp.
- SOH = `100 × measured_discharge_capacity_Ah / rated_capacity_Ah`.
- Preserve each dataset's source EOL rule; use the common 80% threshold as an additional harmonized analysis threshold.
- Never include current-cycle measured capacity/SOH as a predictor.
- Split by whole battery IDs, not random cycle rows.
- Cross-dataset target cells are unseen during source model fitting.
- Few-shot support batteries must be disjoint from target test batteries.
- Oxford characterization checkpoints are sparse; do not interpolate unobserved SOH labels.
- The current shared feature baseline is cycle number only. It is an age baseline, not yet a charging-curve-only estimator.

## Project contents

- `battery_soh/data/`: schema, dataset readers, SOH and feature handling, battery splits
- `battery_soh/models/`: classical and PyTorch regressors
- `battery_soh/experiments/`: within/cross/LODO and few-shot adaptation
- `battery_soh/evaluation/`: metrics and figures
- `scripts/run_classical_cpu.py`: CPU-only benchmark launcher
- `scripts/inspect_downloads.ps1`: local source inspection helper
- `notebooks/colab_deep_learning.ipynb`: Colab deep-learning workflow
- `tests/`: synthetic unit/integration tests
- `reports/REPORT_TEMPLATE.md`: results-writing template

## Limitations still requiring real data

The XJTU parser has been checked against the team's standardized pickle output. The CALCE Excel parser has been checked against a real CS2_33 workbook, and the CADEX parser has been checked against a real CS2_8 text file. MATR cell identity across batch files must be verified. No complete real five-dataset benchmark or final IEEE-style research paper has been produced yet.
