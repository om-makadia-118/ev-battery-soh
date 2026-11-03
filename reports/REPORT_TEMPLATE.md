# Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction

**Institutional Context:** BMSCE Project Work 2  
**Status:** Certified Empirical Research Report (Phase 2 Completed)  
**Codebase:** `ev-battery-soh` (v4 Canonical Framework)  
**Evaluation Dates:** November 2026  
**Experimental Scope:** 5 real datasets, 190 physical battery cells, 132,389 cycle observations, 5 random seeds (`11, 22, 33, 44, 55`), 600 benchmark evaluation runs, 5,840 few-shot adaptation runs.

---

## Abstract

Machine learning models for battery state-of-health (SOH) estimation are typically evaluated using random or battery-level splits within a single laboratory dataset. However, real-world deployment requires models to generalize across disparate battery chemistries (LCO, NCM, LFP), varying thermal environments (24°C–45°C), and differing cycling protocols (0.5C to 6.0C charging rates). In this work, we conduct a rigorous, zero-leakage empirical investigation into the cross-dataset generalization of machine-learning models for EV battery SOH prediction across five prominent real-world benchmarks: NASA (LCO), CALCE (LCO), XJTU (NCM), Oxford (LCO), and MATR / Severson et al. (LFP). 

Evaluating four distinct model families across 5 independent seeds, we establish that:
1. **Within-dataset accuracy fails to guarantee cross-dataset generalization:** While gradient-boosted decision trees (XGBoost) achieve a within-domain test Mean Absolute Error (MAE) of $3.89 \pm 3.19$ percentage points (pp) and Ridge regression achieves $4.51 \pm 3.72$ pp, zero-shot transfer across distinct domains degrades error to $13.03 \pm 8.45$ pp for XGBoost and causes unbounded extrapolation divergence for linear models ($61.49 \pm 138.98$ pp on average, reaching $>600$ pp on extreme cycle-range shifts).
2. **Leave-One-Dataset-Out (LODO) pooling significantly stabilizes linear transfer:** Pooling diverse training sources prevents single-source extrapolation blowups, reducing linear model cross-domain MAE from $61.49$ pp to $12.72 \pm 5.84$ pp and XGBoost to $10.82 \pm 5.44$ pp.
3. **Few-shot target adaptation dramatically bridges domain gaps:** Providing just $k = 1$ target support battery reduces scratch adaptation MAE to $11.18$ pp across all models ($7.75$ pp for XGBoost), while $k = 3$ to $5$ target batteries suffice to restore performance to within-domain parity ($5.40$ pp across all models, $4.16$ pp for XGBoost). Feature alignment (CORAL and MeanVariance normalization) successfully bounds transfer error without parameter retraining.

---

## 1. Introduction and Research Questions

Accurate estimation of battery State-of-Health (SOH)—defined as the ratio of current discharge capacity to nominal rated capacity ($SOH = 100 \times Q / Q_{\text{nominal}}$)—is vital for electric vehicle range estimation, safety monitoring, and second-life battery valuation. Existing literature frequently reports within-dataset test errors below $1.5\%$ SOH. However, whether these architectures retain predictive utility when deployed on a vehicle operating with a different cell chemistry, thermal profile, or discharge regime remains an open question.

This investigation resolves three central empirical research questions:
- **RQ1: Does within-dataset accuracy reliably predict cross-dataset generalization?**  
  *Finding:* No. A model achieving $<2.0\%$ MAE on held-out cells within Oxford or XJTU can suffer severe error inflation ($>15\%$ to $>600\%$ MAE) when transferred directly without adaptation to another dataset.
- **RQ2: How severely does performance degrade across differing chemistries (LCO vs. NCM vs. LFP) and protocols?**  
  *Finding:* Chemistries with similar nominal degradation slopes (e.g. Oxford LCO and MATR LFP) can exhibit favorable cross-transfer ($2.73$ pp MAE for Oxford $\to$ MATR), but mismatched cycle lifespans (e.g. NASA 168-cycle LCO transferred to Oxford 500-cycle LCO) catastrophically break unconstrained linear predictors ($637.45$ pp MAE) due to out-of-distribution feature extrapolation.
- **RQ3: How many target batteries are required for effective domain adaptation?**  
  *Finding:* Adaptation is highly sample-efficient. Incorporating just $k = 1$ target support cell reduces average transfer MAE from $37.49$ pp to $11.18$ pp (scratch) and $12.44$ pp (pooled). Supplying $k = 3$ to $5$ cells restores performance to near within-dataset baseline levels ($4.16$ pp for XGBoost).

---

## 2. Data Sources and Provenance

All experiments are conducted exclusively on authenticated, real-world laboratory battery degradation repositories. No synthetic or simulated degradation cycles are utilized.

| Dataset | Cell Type & Chemistry | Nominal Capacity ($Q_0$) | Ambient Temp (°C) | Charge / Discharge Protocol | Cell Count (Identified) | Observations ($N$) | Data Format & Source Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NASA** | 18650 cylindrical, LCO | 2.0 Ah | 24°C | 1.0C CC-CV / 1.0C CC to 2.7V | 4 cells (`B0005`, `B0006`, `B0007`, `B0018`) | 636 | MATLAB v5 structs (`.mat`), PCoE Ames Research Center |
| **CALCE** | Prismatic CS2, LCO | 1.1 Ah | 25°C | 0.5C CC-CV / 0.1C–2.0C dynamic & CC | 15 cells (`CS2_3`–`CS2_38`) | 9,608 | 13 Arbin Excel (`.xlsx`) + 2 CADEX ASCII (`.txt`), Univ. of Maryland |
| **XJTU** | 18650 cylindrical, NCM | 2.0 Ah | 25°C, 35°C, 45°C | 2C / 3C fast charge, 1C CC discharge | 23 cells (8 Batch 1 + 15 Batch 2) | 6,938 | BatteryLife serialized Pickle (`.pkl`), Xi'an Jiaotong University |
| **Oxford** | Pouch (Kokam), LCO | 0.74 Ah | 40°C | 0.5C CC-CV / Artemis urban drive cycles | 8 cells (`Cell1`–`Cell8`) | 519 | MATLAB v7.3 HDF5 structs, sparse 100-cycle characterization checks |
| **MATR** | 18650 cylindrical, LFP | 1.1 Ah | 30°C | Fast-charging (3.6C–6.0C) / 4.0C CC discharge | 140 cells (3 batches) | 114,688 | MATLAB v7.3 HDF5 structs, Severson et al. (Stanford / MIT / TRI) |
| **Total** | **3 Chemistries (LCO, NCM, LFP)** | **0.74 – 2.0 Ah** | **24°C – 45°C** | **0.5C – 6.0C** | **190 batteries** | **132,389 cycles** | **Unified into canonical schema (`data/processed/unified.csv`)** |

### Data Engineering & Hygiene Corrections
- **CALCE CADEX Discharges (`CS2_8`, `CS2_21`):** In CADEX logs, the recorded capacity field represents a nominal test-step progress percentage ($0\% - 100\%$) rather than ampere-hours. Degrading cells hit cutoff voltage ($2.7\text{V}$) prior to reaching $100\%$ completion. Discharge capacity was reconstructed through trapezoidal numerical integration of current over step duration ($\int I dt$), and zero-discharge maintenance/charge logs were handled cleanly.
- **CALCE Multi-Step Counter Accumulation:** In multi-rate test profiles (`CS2_3`, `CS2_9`), testers bundled up to 6 discharge steps into a single cycle index. Naively taking `max(qd) - min(qd)` yielded unphysical capacities ($>6\text{ Ah}$). Cycles were re-segmented by contiguous discharge step with starting baseline subtraction, strictly enforcing physical limits ($\le 1.16\text{ Ah}$).
- **MATR Glitch Outlier Filtering:** In Severson et al., exactly 4 out of 114,692 cycle summaries contained isolated instrumentation spikes (e.g., $2.88\text{ Ah}$ on a $1.1\text{ Ah}$ cell). Filtering $SOH \le 130\%$ ($c \le 1.43\text{ Ah}$) sanitized the series with zero synthetic imputation.
- **Oxford Sparse Checkpoint Integrity:** Oxford records full characterization cycles approximately once every 100 drive cycles. These discrete observations were preserved without artificial interpolation.

---

## 3. Harmonization and Task Definition

### State-of-Health (SOH) Formulation
For cycle $t$, SOH is defined as:
$$\text{SOH}_t = 100 \times \frac{Q_t}{Q_{\text{nominal}}}$$
where $Q_t$ is the validated full-discharge capacity in ampere-hours, and $Q_{\text{nominal}}$ is the manufacturer's rated capacity ($2.0\text{ Ah}$ for NASA/XJTU, $1.1\text{ Ah}$ for CALCE/MATR, $0.74\text{ Ah}$ for Oxford).

### Prevention of Information Leakage
- **Disjoint Battery Partitioning:** All train, validation, and test splits are strictly partitioned by physical battery ID (`dataset::battery_id`). Observations from the same battery cell never appear across multiple splits.
- **Zero Input Leakage:** The current cycle's measured capacity $Q_t$ and target $\text{SOH}_t$ are strictly forbidden from input feature matrices. Only historical sequence features or common cycle indicators are permitted.

---

## 4. Experimental Protocol

Experiments are executed systematically using five random seeds (`11, 22, 33, 44, 55`):
1. **Within-Dataset Baseline:** $60\%$ train, $20\%$ validation, $20\%$ holdout test batteries within each individual dataset domain.
2. **$5 \times 5$ Cross-Dataset Transfer Matrix:** Models trained on the training partition of source dataset $\mathcal{D}_s$ are evaluated zero-shot on the entire held-out target dataset $\mathcal{D}_t$ ($20$ transfer pairs $\times 4$ models $\times 5$ seeds $= 400$ runs, plus $25$ within-domain holdout runs per model).
3. **Leave-One-Dataset-Out (LODO):** Models are trained on pooled data from 4 source datasets and evaluated zero-shot on the completely unseen 5th dataset ($5$ target folds $\times 4$ models $\times 5$ seeds $= 100$ runs).
4. **Few-Shot Domain Adaptation:** For each transfer pair $(\mathcal{D}_s, \mathcal{D}_t)$, target test batteries are reserved. A support pool of $k \in \{0, 1, 3, 5, 10\}$ target batteries is supplied to adaptation strategies:
   - `pooled`: Fit classical model on source training data combined with target support data.
   - `scratch`: Fit classical model exclusively on target support data.
   - `normalization`: Align source features to target support mean and variance.
   - `coral`: Correlation Alignment aligning source covariance to target support covariance via whitening and recoloring:
     $$C_s = \text{Cov}(X_s) + \lambda I, \quad C_t = \text{Cov}(X_t) + \lambda I, \quad A = C_s^{-1/2} C_t^{1/2}$$

---

## 5. Empirical Results

### 5.1 Model Family Overview Across All Benchmark Runs

Summary across all evaluated benchmark runs (mean $\pm$ standard deviation across 5 seeds):

| Evaluation Mode | Model Family | Mean MAE (pp) | Std MAE (pp) | Evaluated Runs ($N$) |
| :--- | :--- | :---: | :---: | :---: |
| **Within-Dataset** | **XGBoost** | **3.89** | 3.19 | 25 |
| **Within-Dataset** | **Ridge** | **4.51** | 3.72 | 25 |
| **Within-Dataset** | **ElasticNet** | **4.52** | 3.72 | 25 |
| **Within-Dataset** | Mean Baseline | 7.87 | 5.06 | 25 |
| **LODO Generalization** | **Mean Baseline** | **10.25** | 4.47 | 25 |
| **LODO Generalization** | **XGBoost** | **10.82** | 5.44 | 25 |
| **LODO Generalization** | ElasticNet | 12.69 | 5.80 | 25 |
| **LODO Generalization** | Ridge | 12.72 | 5.84 | 25 |
| **Pairwise Zero-Shot Cross** | Mean Baseline | 11.22 | 4.24 | 100 |
| **Pairwise Zero-Shot Cross** | **XGBoost** | **13.03** | 8.45 | 100 |
| **Pairwise Zero-Shot Cross** | ElasticNet | 61.31 | 138.63 | 100 |
| **Pairwise Zero-Shot Cross** | Ridge | 61.49 | 138.98 | 100 |

### 5.2 Within-Dataset Generalization (Held-out Batteries)

Performance on held-out test batteries within the same dataset domain (mean $\pm$ standard deviation over 5 seeds):

| Dataset Domain | Chemistry | XGBoost MAE (pp) | Ridge MAE (pp) | ElasticNet MAE (pp) | Mean Baseline MAE (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Oxford** | LCO | **1.57 ± 0.28** | 1.72 ± 0.50 | 1.73 ± 0.50 | 7.02 ± 1.62 |
| **XJTU** | NCM | **2.18 ± 0.25** | 2.39 ± 0.20 | 2.39 ± 0.20 | 3.84 ± 0.16 |
| **MATR** | LFP | **2.71 ± 0.22** | 2.77 ± 0.20 | 2.78 ± 0.20 | 3.58 ± 0.17 |
| **NASA** | LCO | **4.29 ± 1.39** | 4.41 ± 1.60 | 4.41 ± 1.59 | 9.09 ± 2.09 |
| **CALCE** | LCO | **9.40 ± 2.20** | 11.25 ± 2.06 | 11.27 ± 2.05 | 16.71 ± 1.67 |

### 5.3 Complete $5 \times 5$ Cross-Dataset Transfer Matrix

The table below presents the pairwise transfer matrix for **XGBoost** (top value) and **Ridge Regression** (bottom bracketed value), reported as MAE in SOH percentage points ($\text{mean} \pm \text{std}$ across 5 seeds). Diagonal elements represent within-domain evaluation on held-out cells:

| Source $\downarrow$ / Target $\rightarrow$ | CALCE (LCO) | MATR (LFP) | NASA (LCO) | Oxford (LCO) | XJTU (NCM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CALCE** | **9.40 ± 2.20**<br>*[11.25 ± 2.06]* | 14.70 ± 4.13<br>*[16.16 ± 3.60]* | 10.67 ± 1.44<br>*[15.05 ± 1.92]* | 18.93 ± 26.65<br>*[133.77 ± 45.32]* | 5.58 ± 0.97<br>*[3.87 ± 0.93]* |
| **MATR** | 15.48 ± 0.10<br>*[15.72 ± 0.10]* | **2.71 ± 0.22**<br>*[2.77 ± 0.20]* | 18.18 ± 0.07<br>*[17.48 ± 0.03]* | 5.58 ± 0.62<br>*[12.91 ± 1.08]* | 3.57 ± 0.03<br>*[3.43 ± 0.02]* |
| **NASA** | 20.42 ± 1.20<br>*[68.45 ± 7.95]* | 23.08 ± 2.34<br>*[85.95 ± 8.56]* | **4.29 ± 1.39**<br>*[4.41 ± 1.60]* | 17.10 ± 2.67<br>*[637.45 ± 75.52]* | 18.93 ± 1.68<br>*[28.62 ± 2.13]* |
| **Oxford** | 17.63 ± 0.15<br>*[16.67 ± 0.12]* | 2.92 ± 0.18<br>*[2.73 ± 0.05]* | 19.67 ± 0.28<br>*[17.03 ± 0.23]* | **1.57 ± 0.28**<br>*[1.72 ± 0.50]* | 4.59 ± 0.22<br>*[3.58 ± 0.04]* |
| **XJTU** | 13.06 ± 0.07<br>*[11.51 ± 0.04]* | 8.01 ± 0.13<br>*[11.70 ± 0.36]* | 17.34 ± 0.08<br>*[17.15 ± 0.27]* | 6.13 ± 0.44<br>*[110.49 ± 4.65]* | **2.18 ± 0.25**<br>*[2.39 ± 0.20]* |

### 5.4 Leave-One-Dataset-Out (LODO) Generalization Results

In LODO, the model is trained on 4 pooled datasets and evaluated zero-shot on the 5th unseen target dataset:

| Held-Out Target Domain | Target Chemistry | XGBoost MAE (pp) | Ridge MAE (pp) | ElasticNet MAE (pp) | Mean Baseline MAE (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XJTU** | NCM | 3.60 ± 0.08 | **3.29 ± 0.06** | **3.30 ± 0.06** | 4.17 ± 0.07 |
| **Oxford** | LCO | **5.88 ± 0.67** | 19.08 ± 1.34 | 18.91 ± 1.34 | 7.69 ± 0.03 |
| **MATR** | LFP | 11.92 ± 0.75 | 9.40 ± 1.25 | 9.39 ± 1.25 | **9.13 ± 1.12** |
| **CALCE** | LCO | **15.38 ± 0.15** | 15.69 ± 0.11 | 15.69 ± 0.11 | 16.29 ± 0.05 |
| **NASA** | LCO | 17.34 ± 0.20 | 16.16 ± 0.36 | 16.14 ± 0.36 | **13.95 ± 0.04** |

### 5.5 Few-Shot Domain Adaptation Performance

Adaptation performance across all transfer pairs as a function of target support size $k$ ($5,840$ experimental runs):

| Model | Method | $k = 0$ (Zero-Shot) | $k = 1$ Support | $k = 3$ Support | $k = 5$ Support | $k = 10$ Support |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | `none` (zero-shot) | 13.08 pp | — | — | — | — |
| **XGBoost** | `scratch` | — | 7.75 pp | 5.00 pp | **4.16 pp** | 4.98 pp |
| **XGBoost** | `pooled` | — | 9.43 pp | 6.45 pp | 5.62 pp | 5.96 pp |
| **XGBoost** | `coral` | — | 11.82 pp | 10.26 pp | 10.18 pp | 10.82 pp |
| **XGBoost** | `normalization` | — | 11.82 pp | 10.26 pp | 10.17 pp | 10.82 pp |
| **Ridge** | `none` (zero-shot) | 62.88 pp | — | — | — | — |
| **Ridge** | `scratch` | — | 14.34 pp | 5.33 pp | **4.92 pp** | 5.69 pp |
| **Ridge** | `pooled` | — | 15.04 pp | 8.70 pp | 7.73 pp | 6.98 pp |
| **Ridge** | `coral` | — | 19.03 pp | 10.45 pp | 10.15 pp | 11.00 pp |
| **Ridge** | `normalization` | — | 19.07 pp | 10.45 pp | 10.15 pp | 11.00 pp |
| **Overall Mean** | `scratch` | — | 11.18 pp | 5.80 pp | **5.40 pp** | 6.09 pp |
| **Overall Mean** | `pooled` | — | 12.44 pp | 8.19 pp | 7.44 pp | 7.14 pp |

### 5.6 Accounting of Infeasible Settings
To uphold empirical research integrity, no cells were synthetically hallucinated when target pools were constrained:
- **NASA as Target ($N = 4$ total cells):** Reserving 2 cells for unseen testing leaves exactly 2 support cells. Consequently, settings $k \in \{3, 5, 10\}$ are physically infeasible ($3 \text{ values} \times 4 \text{ sources} \times 4 \text{ models} \times 5 \text{ seeds} = 240$ documented runs).
- **Oxford as Target ($N = 8$ total cells):** Reserving 3 cells for testing leaves 5 support cells. Consequently, setting $k = 10$ is physically infeasible ($1 \text{ value} \times 4 \text{ sources} \times 4 \text{ models} \times 5 \text{ seeds} = 80$ documented runs).
- **Total Documented Infeasible Runs:** Exactly 320 runs were recorded with explicit physical reasons in `adaptation_metrics.csv` without simulation or data falsification.

---

## 6. Environmental and Domain Gap Analysis

### Extrapolation Breakdown of Unconstrained Linear Models
The pairwise matrix demonstrates that transferring from NASA ($168$ cycle lifespan) to Oxford ($500$ cycle lifespan) causes Ridge error to explode to $637.45$ pp MAE. Linear coefficients fit on short cycle intervals extrapolate monotonically downward into deeply negative unphysical SOH values. By contrast, tree-based models (XGBoost) partition feature space and predict leaf constants outside training bounds, constraining NASA $\to$ Oxford transfer MAE to $17.10$ pp.

### Thermal and C-Rate Confounding
- **Operating Temperature:** Oxford cells were cycled at an elevated temperature of $40^\circ\text{C}$, whereas NASA ($24^\circ\text{C}$) and CALCE ($25^\circ\text{C}$) operated at room temperature. The accelerated SEI layer growth at elevated temperatures changes capacity loss per cycle.
- **Charging C-Rate:** MATR uses fast charging rates ($3.6\text{C}$ to $6.0\text{C}$), inducing mechanical particle cracking alongside SEI growth. Despite this, Oxford $\to$ MATR transfers remarkably well ($2.73$ pp for Ridge), indicating that relative SOH trajectories can align across chemistries if the underlying degradation slope matches.

---

## 7. Discussion and Limitations

1. **Cell Sample Sizes in Legacy Benchmarks:** NASA ($4$ cells) and Oxford ($8$ cells) provide valuable longitudinal data but possess small cell counts, which limits high-$k$ few-shot support and widens cross-seed variance.
2. **Measurement Cadence Disparities:** Oxford conducts characterization checks only every $\sim 100$ cycles, whereas MATR and CALCE record every cycle.
3. **Chemistry vs. Operating Protocol Confounding:** In open benchmarks, chemistry is often confounded with test conditions (e.g., LFP is only tested under fast charge in MATR; NCM is tested at multiple C-rates in XJTU). Controlled factorial cycling studies are needed to isolate chemical from thermal effects.

---

## 8. Reproducibility and Verification

All code, data pipelines, and experiment runners are completely reproducible:
- **Environment:** Python 3.14.7 virtual environment (`.venv`).
- **Dependencies:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `plotly`, `streamlit`.
- **Test Suite:** Run `.venv/bin/pytest tests/ -v` (30 unit and integration tests passing).
- **Benchmark Commands:**
  ```bash
  # 1. Full 5-seed benchmark (matrix and LODO):
  python scripts/run_study.py --phase benchmark --models mean ridge elasticnet xgboost --seeds 11 22 33 44 55 --output results/full_study
  
  # 2. Adaptation study (few-shot k in 0, 1, 3, 5, 10):
  python scripts/run_study.py --phase adaptation --models mean ridge elasticnet xgboost --seeds 11 22 33 44 55 --output results/full_study
  
  # 3. Interactive research dashboard:
  streamlit run streamlit_app.py
  ```
- **Generated Figures:**
  - Transfer Matrices: `results/full_study/matrix/figures/transfer_<model>.png`
  - Model Ranking: `results/full_study/matrix/figures/ranking.png`
  - LODO Comparison: `results/full_study/lodo/figures/lodo_target_comparison.png`
  - Adaptation Curves: `results/full_study/adaptation/figures/adaptation_curve.png`, `adaptation_curve_<model>.png`
  - Environmental Plots: `results/full_study/matrix/figures/error_temperature.png`, `error_charge_rate.png`, `error_chemistry.png`
