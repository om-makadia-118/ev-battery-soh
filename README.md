# Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction

[![Tests](https://img.shields.io/badge/tests-30%20passed%2C%201%20skipped-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue)](pyproject.toml)
[![Framework](https://img.shields.io/badge/framework-v4%20Certified-teal)](battery_soh/)
[![Benchmark](https://img.shields.io/badge/benchmark-6%2C440%20runs%20(5%20seeds)-orange)](results/full_study/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Institutional Context:** BMS College of Engineering (BMSCE) — Project Work 2 (Phase 2 Completed)  
**Author:** Divyam Jain  
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

Evaluating **190 physical battery cells** and **132,389 validated cycle observations** across four model architectures and five random seeds (`11, 22, 33, 44, 55`), we executed 6,440 total benchmark and adaptation runs with zero synthetic data.

---

## 🔬 Core Empirical Findings

| Evaluation Protocol | XGBoost MAE (pp) | Ridge Regression (pp) | ElasticNet (pp) | Naive Mean Baseline (pp) | Evaluated Runs ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Within-Dataset Holdout** | **3.89 ± 3.19** | **4.51 ± 3.72** | 4.52 ± 3.72 | 7.87 ± 5.06 | 25 |
| **LODO Generalization** | **10.82 ± 5.44** | 12.72 ± 5.84 | 12.69 ± 5.80 | 10.25 ± 4.47 | 25 |
| **Zero-Shot Pairwise Transfer** | **13.03 ± 8.45** | 61.49 ± 138.98 | 61.31 ± 138.63 | 11.22 ± 4.24 | 100 |

1. **Within-Domain Accuracy Does Not Transfer Zero-Shot:** While XGBoost achieves $3.89 \pm 3.19$ percentage points (pp) MAE on held-out cells of the same domain, zero-shot transfer degrades error to $13.03 \pm 8.45$ pp.
2. **Linear Models Suffer Catastrophic Extrapolation Divergence:** When transferring from short cycle lifespans (NASA 168 cycles) to longer lifespans (Oxford 520 cycles), unconstrained linear models extrapolate into deeply negative unphysical SOH ($637.45 \pm 75.52$ pp MAE), whereas tree-based ensembles (XGBoost) remain bounded ($17.10 \pm 2.67$ pp).
3. **Multi-Domain Pooling (LODO) Tames Linear Instability:** In Leave-One-Dataset-Out evaluation, training on four pooled domains eliminates single-source blowups, pulling Ridge error down to $12.72 \pm 5.84$ pp.
4. **Few-Shot Adaptation is Highly Sample-Efficient:** Supplying just $k = 1$ target support battery slashes transfer MAE from $37.49$ pp to $11.18$ pp across all models ($7.75$ pp for XGBoost), while $k = 3$ to $5$ target cells restore accuracy to within-domain baseline parity ($4.16$ pp for XGBoost).

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
├── scripts/                   # Command-line benchmark launchers
│   ├── run_classical_cpu.py   # Quick CPU pilot launcher
│   └── run_study.py           # Multi-phase 5-seed benchmark runner
├── tests/                     # Unit and integration test suite (30 passing tests)
├── .streamlit/                # Streamlit theme and server configuration
├── pyproject.toml             # Package metadata and build configuration
├── requirements.txt           # Python package dependencies
├── streamlit_app.py           # Presentation dashboard & live what-if simulator
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

## 📜 Citation

If you find this benchmark or codebase useful in your research, please cite:

```bibtex
@misc{jain2026evbatterysoh,
  author = {Jain, Divyam},
  title = {Cross-Dataset Generalization and Few-Shot Adaptation of Machine-Learning Models for Electric Vehicle Battery State-of-Health Estimation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/om-makadia-118/ev-battery-soh}}
}
```
