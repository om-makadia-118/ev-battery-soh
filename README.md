# Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction

[![Tests](https://img.shields.io/badge/tests-30%20passed%2C%201%20skipped-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue)](pyproject.toml)
[![Framework](https://img.shields.io/badge/framework-v4%20Certified-teal)](battery_soh/)
[![Benchmark](https://img.shields.io/badge/benchmark-6%2C440%20runs%20(5%20seeds)-orange)](results/full_study/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Institutional Context:** BMS College of Engineering (BMSCE) — Project Work 2 (Phase 2 Completed)  
**Authors:** Om Manish Makadia, Divyam Jain, Goutham T. G., D. Pranav Sai, M. Dakshayini (project guide)  
**Department:** Artificial Intelligence and Machine Learning, BMSCE  
**Repository:** [https://github.com/om-makadia-118/ev-battery-soh](https://github.com/om-makadia-118/ev-battery-soh)

---

## 📌 Executive Summary

State-of-Health (SOH) estimation is critical for electric vehicle (EV) safety, range estimation, and battery retirement. However, conventional data-driven research predominantly evaluates models on homogeneous laboratory batches with random cycle splitting—masking severe domain shift vulnerabilities.

This project delivers an end-to-end, zero-leakage empirical benchmark investigating **cross-dataset generalization and few-shot domain adaptation** across five prominent open-access battery repositories:
- **NASA Ames PCoE** ($\text{LiCoO}_2$, 2.0 Ah, 24°C, 4 cells)
- **University of Maryland CALCE** ($\text{LiCoO}_2$ CS2, 1.1 Ah, 25°C, 15 cells)
- **Xi'an Jiaotong University (XJTU)** ($\text{LiNiCoMnO}_2$, 2.0 Ah, 25°C–45°C, 23 cells)
- **University of Oxford** ($\text{LiCoO}_2$ pouch, 0.74 Ah, 40°C, 8 cells)
- **MIT-Stanford-Toyota (MATR / Severson et al.)** ($\text{LiFePO}_4$, 1.1 Ah, 30°C, 140 cells)

Evaluating **190 physical battery cells** and **132,389 validated cycle observations** across four model architectures and five random seeds (`11, 22, 33, 44, 55`), we logged 6,440 benchmark and adaptation runs (6,120 executed plus 320 few-shot settings that were infeasible because the target dataset has too few cells), with zero synthetic data. The only predictor is the cycle index, the one input available in all five datasets; see the paper for what this does and does not show.

---

## 🔬 Core Empirical Findings

Lowest MAE per row in bold. Splits are always by whole cell; the only predictor is the cycle index.

| Evaluation Protocol | XGBoost MAE (pp) | Ridge Regression (pp) | ElasticNet (pp) | Naive Mean Baseline (pp) | Evaluated Runs ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Within-Dataset Holdout** | **3.89 ± 3.19** | 4.51 ± 3.72 | 4.52 ± 3.72 | 7.87 ± 5.06 | 25 |
| **LODO Generalization** | 10.82 ± 5.44 | 12.72 ± 5.84 | 12.69 ± 5.80 | **10.25 ± 4.47** | 25 |
| **Zero-Shot Pairwise Transfer** | 13.03 ± 8.45 | 61.49 ± 138.98 | 61.31 ± 138.63 | **11.22 ± 4.24** | 100 |

1. **Within-dataset accuracy does not transfer zero-shot, and it does not beat a trivial baseline:** XGBoost reaches $3.89 \pm 3.19$ pp MAE on held-out cells of the same dataset, but zero-shot transfer degrades to $13.03 \pm 8.45$ pp. The training-mean baseline gets $11.22 \pm 4.24$ pp; XGBoost beats it in only 8 of 20 source-target pairs.
2. **Linear models diverge when the target cycle range exceeds the source range:** NASA cells end at 168 cycles and Oxford at 8,200, so Ridge extrapolates to unphysical SOH ($637.45 \pm 75.52$ pp MAE for NASA to Oxford), whereas XGBoost stays bounded ($17.10 \pm 2.67$ pp). Across the 20 pairs, Ridge error correlates with the target/source cycle-range ratio (Spearman 0.55).
3. **Pooling (LODO) bounds the error but does not beat the mean baseline:** LODO gives $10.82 \pm 5.44$ pp for XGBoost versus $10.25 \pm 4.47$ pp for the training mean. MATR supplies 87% of all rows, so pooled training is dominated by MATR.
4. **Target labels help from three support cells; label-free alignment does not:** On the three targets feasible at every $k$ (CALCE, MATR, XJTU), XGBoost trained on the support cells reaches 6.22, 5.12 and 4.98 pp at $k = 3, 5, 10$ versus about 8 pp for the target-mean predictor. With $k = 1$ nothing beats the target mean (8.53 pp). CORAL and mean-variance normalisation stay at 10.8-12.9 pp.

**Limits:** cycle index is the only predictor; CALCE labels are noisy (12.5% of observations below 50% SOH); NASA has 4 cells and Oxford 8. Results without CALCE give the same ordering. See `paper/paper.pdf`.

---

## 🚀 Quick Start & Installation

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/om-makadia-118/ev-battery-soh.git
cd ev-battery-soh

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On macOS/Linux
# .venv\Scripts\activate        # On Windows PowerShell

# Install dependencies in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
```

### 2. Verify the Test Suite
Run the 30-test unit and integration test suite:
```bash
pytest tests/ -v
# Output: 30 passed, 1 skipped in ~1.5s
```

### 3. Launch the Interactive Research Dashboard
Launch the presentation dashboard to explore interactive heatmaps, LODO charts, adaptation curves, and the live SOH inference simulator:
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 Running the Benchmarks

### 1. Classical Matrix & LODO Benchmark (CPU)
Run the 5-seed benchmark across all four classical models:
```bash
python scripts/run_study.py --phase benchmark --models mean ridge elasticnet xgboost --seeds 11 22 33 44 55 --output results/full_study
```

### 2. Few-Shot Domain Adaptation Benchmark
Run few-shot adaptation ($k \in \{0, 1, 3, 5, 10\}$) across all 20 source-target domain pairs:
```bash
python scripts/run_study.py --phase adaptation --models mean ridge elasticnet xgboost --seeds 11 22 33 44 55 --output results/full_study
```

### 3. Deep Learning Sequence Models (Google Colab GPU)
For sequence models (LSTM, 1D-CNN, Transformer):
1. Open [`notebooks/colab_deep_learning.ipynb`](notebooks/colab_deep_learning.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Upload `data/processed/unified.csv` and run cells in sequence.

---

## 📁 Repository Structure

```
ev-battery-soh/
├── battery_soh/               # Core Python library (v4 canonical framework)
│   ├── data/                  # Schema harmonization, loaders, and disjoint splits
│   │   ├── features.py        # Safe feature extraction (zero target leakage)
│   │   ├── loaders.py         # Native parsers: NASA, CALCE, XJTU, Oxford, MATR
│   │   ├── schema.py          # SOH definition and physical bounds validation
│   │   └── splits.py          # Strict battery-disjoint train/val/test splitting
│   ├── evaluation/            # Evaluation metrics and figure generation
│   │   ├── metrics.py         # MAE, RMSE, MAPE regression metrics
│   │   └── plots.py           # Heatmaps, rankings, LODO, and adaptation curves
│   ├── experiments/           # Benchmark and adaptation execution engines
│   │   ├── adaptation.py      # Few-shot adaptation (scratch, pooled, coral, norm)
│   │   └── engine.py          # Matrix, LODO, and within-domain runners
│   └── models/                # Model implementations
│       ├── classical.py       # Mean, Ridge, ElasticNet, XGBoost wrappers
│       └── deep.py            # PyTorch LSTM, 1D-CNN, Transformer regressors
├── data/                      # Data storage (raw data ignored by git)
│   └── processed/             # Unified dataset (unified.csv, data_quality.csv)
├── docs/                      # Technical documentation & project handoff notes
│   ├── IMPLEMENTATION_STATUS.md # Certified implementation checklist
│   ├── DATA_LICENSES.md       # Dataset sources, citations and licence notes
│   ├── LLM_PROJECT_CONTEXT.md # Full project guidelines and audit constraints
│   └── RESEARCH_PAPER_REFERENCE.md # IEEE format research reference
├── notebooks/                 # GPU-accelerated deep learning
│   └── colab_deep_learning.ipynb # Colab GPU notebook for sequence models
├── reports/                   # Publication-grade reports and academic manuscripts
│   ├── PROJECT_REPORT.md      # Comprehensive academic project report (BMSCE)
│   ├── REPORT_TEMPLATE.md     # Synchronized project report
│   └── RESEARCH_PAPER_REFERENCE.md # IEEE format research paper blueprint
├── results/                   # Generated benchmark outputs
│   └── full_study/            # Certified 5-seed benchmark results and figures
│       ├── adaptation/        # Few-shot learning curves & metrics
│       ├── lodo/              # LODO generalization bars & rankings
│       └── matrix/            # 5x5 pairwise transfer heatmaps & CSVs
├── paper/                     # IEEE-format manuscript (LaTeX), references, figures, compiled PDF
├── scripts/                   # Command-line benchmark launchers
│   ├── make_paper_assets.py   # Regenerates every paper table/figure from results + unified.csv
│   ├── run_classical_cpu.py   # Quick CPU pilot launcher
│   └── run_study.py           # Multi-phase 5-seed benchmark runner
├── tests/                     # Unit and integration test suite (30 passing tests)
├── .streamlit/                # Streamlit theme and server configuration
├── pyproject.toml             # Package metadata and build configuration
├── requirements.txt           # Python package dependencies
├── streamlit_app.py           # Presentation dashboard & live what-if simulator
├── LICENSE                    # MIT licence (code)
├── CITATION.cff               # Citation metadata
└── README.md                  # This document
```

---

## 🛡️ Scientific Integrity & Research Safeguards

- **Strict Battery-Disjoint Splitting:** Splits partition exclusively by physical cell ID (`battery_split`). Observations from the same cell never cross partitions.
- **Zero Target Feature Leakage:** Current-cycle capacity $Q_t$ and target $\text{SOH}_t$ are strictly forbidden from predictors.
- **Zero Synthetic Experiments:** All benchmark scores are derived exclusively from actual raw laboratory files.
- **No Label Interpolation:** Oxford discrete characterization checkpoints (~100 cycles) are preserved without artificial interpolation.
- **Explicit Infeasibility Accounting:** Exactly 320 infeasible few-shot settings (NASA $k \ge 3$, Oxford $k = 10$) are logged in `adaptation_metrics.csv` due to real cell pool bounds; zero cells were fabricated.

---

## 📖 Key Documentation & Academic Reference

- **IEEE Research Paper Reference:** [`reports/RESEARCH_PAPER_REFERENCE.md`](reports/RESEARCH_PAPER_REFERENCE.md)
- **Comprehensive Academic Project Report:** [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md)
- **Implementation Status Audit:** [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)

---

## 📄 Paper

The manuscript (IEEE format, LaTeX) is in [`paper/`](paper/): source `paper.tex`, bibliography `refs.bib`, figures `paper/figs/`, compiled `paper/paper.pdf`.

```bash
python scripts/make_paper_assets.py     # regenerate figures, tables and paper/numbers.json
cd paper && latexmk -pdf paper.tex      # rebuild the PDF
```

## ⚖️ License and data

Code is released under the MIT licence ([`LICENSE`](LICENSE)). The five source datasets keep their own licences and must be cited; raw files (about 11.7 GB) are not redistributed. See [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for download links and licence notes.

## 📜 Citation

If you find this benchmark or codebase useful in your research, please cite:

```bibtex
@misc{makadia2026evbatterysoh,
  author = {Makadia, Om Manish and Jain, Divyam and T. G., Goutham and Sai, D. Pranav and Dakshayini, M.},
  title = {Cross-Dataset Generalization and Few-Shot Adaptation of Machine-Learning Models for Electric Vehicle Battery State-of-Health Estimation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/om-makadia-118/ev-battery-soh}}
}
```
