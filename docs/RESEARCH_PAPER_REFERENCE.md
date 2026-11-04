# Cross-Dataset Generalization and Few-Shot Adaptation of Machine-Learning Models for Electric Vehicle Battery State-of-Health Estimation

**Divyam Jain**  
*Department of Computer Science and Engineering*  
*B.M.S. College of Engineering (BMSCE), Autonomous Institute Affiliated to VTU*  
*Bangalore, Karnataka, India*  

---

### Abstract
State-of-Health (SOH) estimation is critical for safe, optimal, and reliable battery management in electric vehicles (EVs). While recent data-driven models report within-dataset Mean Absolute Errors (MAE) below 1.5% SOH, these evaluations almost universally rely on homogeneous laboratory data. In real-world vehicular fleets, battery management systems (BMS) must generalize across differing cathode chemistries ($\text{LiCoO}_2$, $\text{LiNiCoMnO}_2$, $\text{LiFePO}_4$), fluctuating ambient temperatures (24°C–45°C), dynamic driving schedules, and disparate charging protocols (0.5C to 6.0C). This paper presents a comprehensive, zero-leakage empirical benchmark investigating the cross-dataset generalization of machine-learning models across five landmark battery degradation repositories: NASA, CALCE, XJTU, Oxford, and MATR (Severson et al.), comprising 190 physical battery cells and 132,389 validated cycling observations. Evaluating four distinct model families across five independent random seeds (6,440 total benchmark and adaptation runs), we establish three primary empirical contributions: (1) within-domain accuracy fails to predict cross-domain transferability—Gradient Boosted Decision Trees (XGBoost) achieve $3.89 \pm 3.19$ percentage points (pp) MAE on held-out cells within the same domain, but zero-shot cross-dataset transfer degrades error to $13.03 \pm 8.45$ pp for XGBoost and triggers catastrophic extrapolation failure for unconstrained linear models ($61.49 \pm 138.98$ pp average MAE, exceeding 600 pp on cycle-range mismatches); (2) Leave-One-Dataset-Out (LODO) multi-domain pooling serves as an effective regularizer, reducing linear model cross-domain MAE to $12.72 \pm 5.84$ pp and XGBoost to $10.82 \pm 5.44$ pp; and (3) few-shot target adaptation is exceptionally sample-efficient—incorporating just $k = 1$ target support battery reduces scratch adaptation MAE to $11.18$ pp across all models ($7.75$ pp for XGBoost), while $k = 3$ to $5$ support batteries restore performance to within-domain parity ($4.16$ pp for XGBoost).

*Index Terms*—Battery management systems, cross-dataset generalization, domain adaptation, few-shot learning, lithium-ion batteries, machine learning, state of health (SOH), transfer learning.

---

## I. INTRODUCTION

LITHIUM-ION batteries represent the dominant energy storage technology for electric vehicles (EVs) and smart grids due to their high specific energy, high power capability, and falling unit costs. However, during real-world vehicular operation, battery cells undergo progressive, irreversible degradation driven by parasitic chemical side-reactions, including Solid Electrolyte Interphase (SEI) layer growth at the negative electrode, loss of cyclable lithium inventory, active material dissolution, and lithium plating during high-rate charging [1], [2]. Because internal chemical states cannot be monitored directly via non-invasive on-board sensors, the Battery Management System (BMS) relies on algorithmic estimators to infer State-of-Health (SOH).

Accurate SOH estimation is critical for determining vehicle driving range, preventing thermal runaway, tailoring charging speed, and appraising battery residual value for second-life grid applications [3]. Driven by the availability of public cycling datasets, data-driven machine learning (ML) and deep learning (DL) models have gained immense popularity, frequently reporting within-dataset estimation errors under $1.5\%$ SOH [4], [5].

Despite these encouraging laboratory numbers, conventional evaluation methodologies suffer from three fundamental flaws:
1. *Random Row Partitioning and Temporal Contamination:* Researchers frequently shuffle cycle rows from the same battery cell across training and test splits. Because adjacent cycles of a degrading cell exhibit strong temporal autocorrelation, random cycle splitting leaks future degradation trajectory data, producing artificially optimistic test errors that fail to replicate in production [6].
2. *Homogeneous Laboratory Overfitting:* Models are almost universally trained and tested on cells from a single manufacturing batch, cycled at a fixed laboratory temperature, under identical test protocols. In automotive fleets, batteries encounter dramatic domain shifts in chemistry, ambient climate, dynamic drive schedules, and charging C-rates.
3. *Target Feature Leakage:* Several published pipelines utilize features derived from current-cycle discharge capacity or end-of-discharge voltage curves to estimate current-cycle health, leaking the prediction target into the input feature matrix.

To address these critical limitations, this work establishes an exhaustive, zero-leakage empirical benchmark evaluating the cross-dataset generalization of machine-learning models across five major public battery repositories: NASA Ames PCoE [2], University of Maryland CALCE [3], Xi'an Jiaotong University (XJTU) [5], Oxford University [4], and MIT-Stanford-Toyota (MATR / Severson et al.) [1], totaling 190 physical battery cells and 132,389 observed cycles.

The primary contributions of this paper are:
- **Zero-Leakage Benchmark Pipeline:** We formulate and implement strict battery-disjoint partitioning ($\mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{test}} = \emptyset$) and enforce a zero-feature-leakage policy across all five heterogeneous repositories.
- **Complete $5 \times 5$ Cross-Dataset Transfer Matrix:** We evaluate pairwise zero-shot transfer across 20 distinct source-target domain pairs and 5 within-domain diagonals across four model architectures and five random seeds (500 runs).
- **Multi-Domain Pooling via LODO:** We quantify the regularizing capability of Leave-One-Dataset-Out (LODO) training across four pooled domains evaluated on a completely unseen fifth domain (100 runs).
- **Sample-Efficient Few-Shot Domain Adaptation:** We investigate the sample efficiency of target pack adaptation across support sizes $k \in \{0, 1, 3, 5, 10\}$ using scratch training, pooled learning, MeanVariance normalization, and Correlation Alignment (CORAL) (5,840 runs), providing explicit accounting of physical cell pool limits.
- **Diagnostic Electrochemical Breakdown:** We analyze the mathematical and electrochemical drivers of domain shift, identifying why unconstrained linear estimators experience catastrophic extrapolation divergence on cycle-lifespan mismatches while tree-based ensembles remain bounded.

---

## II. RELATED WORK AND PROBLEM FORMULATION

### A. State-of-Health Estimation Methodologies
Conventional SOH estimation techniques in BMS include Coulomb counting (current integration), open-circuit voltage (OCV) relaxation mapping, and Equivalent Circuit Model (ECM) parameter tracking via Extended Kalman Filters (EKF) [7]. While Coulomb counting is computationally lightweight, it is susceptible to current sensor drift and requires periodic recalibration at full charge/discharge [8]. OCV relaxation requires extensive resting periods (often 1–2 hours) for electrochemical equilibrium, which is impractical in moving vehicles.

Data-driven ML approaches model the mapping from operational cycle features $X_t$ to capacity-derived SOH directly. Commonly deployed techniques include Support Vector Regression (SVR), Gaussian Process Regression (GPR), Gradient Boosted Decision Trees (XGBoost), Long Short-Term Memory (LSTM) networks, and Transformers [4], [5], [9]. However, existing literature overwhelmingly reports within-dataset accuracy, leaving cross-dataset transferability largely uncharacterized.

### B. Mathematical Definition of State-of-Health
State-of-Health ($\text{SOH}_t$) at operational cycle $t$ is defined as the ratio of validated full discharge capacity $Q_t$ to manufacturer nominal rated capacity $Q_{\text{nominal}}$:
$$\text{SOH}_t = \frac{Q_t}{Q_{\text{nominal}}} \times 100 \quad [\%] \tag{1}$$
Where $Q_{\text{nominal}}$ is fixed by cell manufacturer specifications: $2.0\text{ Ah}$ for NASA and XJTU, $1.1\text{ Ah}$ for CALCE and MATR, and $0.74\text{ Ah}$ for Oxford.

### C. End-of-Life (EOL) Standards
In industrial standards, End-of-Life (EOL) is reached when available capacity degrades below a predefined threshold:
- *Source-Specific EOL:* Preserved per original research protocols: NASA (70%), CALCE (70%), XJTU (80%), Oxford ($\approx 80\%$), MATR (80%).
- *Harmonized Benchmark EOL:* Fixed at 80.0% SOH across all datasets to enable standardized multi-chemistry comparison. Post-EOL cycles are retained in the feature space to evaluate non-linear degradation knee behavior.

---

## III. MULTI-DATASET BENCHMARK REPOSITORIES AND PROVENANCE

The benchmark investigates five authenticated open-access repositories covering 190 physical cells and 132,389 validated cycle observations, summarized in Table I.

```
+--------------------------------------------------------------------------------------------------+
| TABLE I: SPECIFICATIONS OF BENCHMARK BATTERY DATASETS                                            |
+----------+-----------------+-------------+-------------+-------+----------+----------------------+
| Dataset  | Positive / Neg. | Form Factor | Nominal Cap | Temp. | Cells    | Charge / Discharge   |
| Name     | Electrode       |             | (Ah)        | (°C)  | Count    | Protocol             |
+----------+-----------------+-------------+-------------+-------+----------+----------------------+
| NASA     | LCO / Graphite  | 18650 Cyl.  | 2.0         | 24    | 4 cells  | 1.0C CC-CV / 1.0C CC |
| CALCE    | LCO / Graphite  | CS2 Prism.  | 1.1         | 25    | 15 cells | 0.5C / 0.1C–2.0C dyn |
| XJTU     | NCM / Graphite  | 18650 Cyl.  | 2.0         | 25-45 | 23 cells | 2C, 3C / 1.0C CC     |
| Oxford   | LCO / Graphite  | Pouch SLPB  | 0.74        | 40    | 8 cells  | 0.5C / Artemis dyn   |
| MATR     | LFP / Graphite  | 18650 Cyl.  | 1.1         | 30    | 140 cell | 3.6C-6C / 4.0C CC    |
+----------+-----------------+-------------+-------------+-------+----------+----------------------+
```

### A. NASA Ames PCoE Repository
The NASA PCoE battery dataset [2] consists of commercial 18650-size lithium cobalt oxide ($\text{LiCoO}_2$, LCO) cells cycled at room temperature (24°C). The four fully validated cells (`B0005`, `B0006`, `B0007`, `B0018`) underwent 1.0C CC-CV charging to 4.2V and 1.0C (2.0A) CC discharging to 2.7V. Cycling continues until reaching approximately 70% SOH over 132 to 168 cycles.

### B. CALCE CS2 Repository
The University of Maryland Center for Advanced Life Cycle Engineering (CALCE) CS2 dataset [3] comprises prismatic LCO cells rated at 1.1 Ah, cycled at 25°C under varying constant and dynamic discharge rates (0.1C to 2.0C) down to 2.7V. Fifteen physical cells (`CS2_3` through `CS2_38`) were ingested from 13 Arbin Excel workbooks and 2 CADEX text archives.

### C. XJTU BatteryLife Repository
The Xi'an Jiaotong University dataset [5] evaluates commercial 18650 cylindrical cells with nickel-cobalt-manganese ($\text{LiNiCoMnO}_2$, NCM) chemistry rated at 2.0 Ah. Cycling was conducted across ambient temperatures of 25°C, 35°C, and 45°C under aggressive fast-charging conditions (2.0C in Batch 1; 3.0C in Batch 2) with 1.0C discharge to 2.5V, covering 23 physical cells and 6,938 cycle observations.

### D. Oxford Battery Degradation Dataset
The Oxford repository [4] evaluates 8 commercial Kokam SLPB 740 mAh lithium cobalt oxide pouch cells cycled at an elevated ambient temperature of 40°C. Cells undergo dynamic Artemis urban driving schedules, with standardized 1C capacity characterization checks conducted every $\sim 100$ drive cycles over 519 discrete checkpoint cycles.

### E. MATR (Severson et al.) Dataset
The MIT-Stanford-Toyota collection [1] is the largest public battery degradation repository, evaluating commercial A123 Systems APR18650M1A lithium iron phosphate ($\text{LiFePO}_4$, LFP) cells rated at 1.1 Ah. The 140 cells across three batches were tested at 30°C under multi-step fast-charging regimes (ranging from 3.6C to 6.0C) with 4.0C discharge to 2.0V, encompassing 114,688 validated cycles.

---

## IV. DATA PREPROCESSING AND ZERO-LEAKAGE HARMONIZATION

### A. Raw Archive Parser Remediation
To guarantee data fidelity, Phase 1 corrected four critical parser defects identified in the raw archives:
1. *CALCE CADEX Numerical Integration:* In `CS2_8` and `CS2_21`, the recorded `Capacity` field represents progress percentage ($0\% - 100\%$) rather than ampere-hours. Degrading cells hit the 2.7V cutoff prior to reaching 100% completion. True capacity was reconstructed via trapezoidal numerical integration of discharge current over step duration:
   $$Q_{\text{dis}} = \int_{t_{\text{start}}}^{t_{\text{end}}} |I(t)| \, dt = \sum_{k=1}^{N-1} \frac{|I_k| + |I_{k+1}|}{2} \Delta t_k \tag{2}$$
   Physical cutoff filters ($V_{\min} \le 3.0\text{V}$, $V_{\max} \ge 3.8\text{V}$, duration $\ge 300\text{s}$, $Q_{\text{dis}} \ge 0.05\text{ Ah}$) were enforced.
2. *CALCE Arbin Step Segmentation:* In multi-rate test profiles (`CS2_3`, `CS2_9`), Arbin equipment bundled up to six discharge steps into a single cycle index. Re-segmenting by `_dis_step` with starting baseline subtraction prevented cumulative counter accumulation, capping capacities at physical boundaries ($\le 1.16\text{ Ah}$).
3. *MATR Hardware Glitch Outlier Filtering:* Exactly 4 single-cycle measurements out of 114,692 rows exhibited unphysical hardware spikes ($>1.43\text{ Ah}$). Enforcing physical bounds ($SOH \le 130\%$) sanitized the series without artificial smoothing.
4. *Oxford Sparse Checkpoint Integrity:* Discrete characterization checkpoints (~100 cycles) were preserved without artificial interpolation.

### B. Strict Battery-Disjoint Splitting
To prevent temporal contamination, all data splits partition by unique battery identifier (`dataset::battery_id`):
$$\mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{val}} = \emptyset, \quad \mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{test}} = \emptyset, \quad \mathcal{B}_{\text{val}} \cap \mathcal{B}_{\text{test}} = \emptyset \tag{3}$$
All observations belonging to battery cell $i$ reside exclusively within one assigned partition.

### C. Zero-Leakage Feature Formulation
The input feature matrix $X_t$ consists solely of the operational cycle index:
$$X_t = [c_t] \tag{4}$$
The current cycle's measured discharge capacity $Q_t$ and target $\text{SOH}_t$ are strictly forbidden from $X_t$.

---

## V. MODEL ARCHITECTURES AND DOMAIN ALIGNMENT

### A. Evaluated Model Architectures
1. *Naive Mean Baseline:*
   $$\hat{y}_t = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i \tag{5}$$
   Serves as the empirical minimum reference; any practical model must substantially outperform the training mean.
2. *Ridge Regression (L2 Regularization):*
   $$\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2 \implies w = (X^T X + \alpha I)^{-1} X^T y \tag{6}$$
   Where $\alpha = 1.0$. Fits a regularized linear degradation slope.
3. *ElasticNet Regression (L1 + L2 Regularization):*
   $$\min_w \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right) \tag{7}$$
   Where $\alpha = 0.1$ and $\rho = 0.5$ (equal balance between Lasso and Ridge penalties).
4. *XGBoost (Gradient Boosted Decision Trees):*
   Ensemble of $K$ additive regression trees:
   $$\hat{y}_i = \sum_{k=1}^K f_k(X_i), \quad \mathcal{L} = \sum_{i=1}^N l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k) \tag{8}$$
   Where $\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \sum_{j=1}^{T_k} w_j^2$. Hyperparameters: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`.

### B. Few-Shot Domain Adaptation Formulations
When adapting a source model using $k$ target support batteries $\mathcal{D}_{t,\text{support}}$:
1. *Scratch Adaptation:* Model is trained exclusively on target support cells:
   $$w^* = \arg\min_w \mathcal{L}(w; \mathcal{D}_{t,\text{support}}) \tag{9}$$
2. *Pooled Adaptation:* Source training data is concatenated with target support data:
   $$w^* = \arg\min_w \mathcal{L}(w; \mathcal{D}_s \cup \mathcal{D}_{t,\text{support}}) \tag{10}$$
3. *MeanVariance Normalization Alignment:*
   Aligns source features to target support mean and variance without modifying model weights:
   $$\hat{X}_s = \frac{X_s - \mu_s}{\sigma_s} \cdot \sigma_{t,\text{support}} + \mu_{t,\text{support}} \tag{11}$$
4. *Correlation Alignment (CORAL):*
   Minimizes domain covariance distance by whitening the source feature covariance and recoloring with target support covariance:
   $$C_s = \text{Cov}(X_s) + \lambda I, \quad C_t = \text{Cov}(X_{t,\text{support}}) + \lambda I \tag{12}$$
   $$A = C_s^{-1/2} C_t^{1/2}, \quad \hat{X}_s = (X_s - \mu_s) A + \mu_{t,\text{support}} \tag{13}$$
   Where $C^{-1/2} = U \Lambda^{-1/2} U^T$ is calculated via symmetric eigendecomposition.

---

## VI. EXPERIMENTAL SETUP AND PROTOCOLS

```
+--------------------------------------------------------------------------------------------------+
| FIG. 1: FOUR BENCHMARK EVALUATION PROTOCOLS                                                      |
+--------------------------+--------------------------+--------------------------------------------+
| Protocol 1: Within       | Protocol 2: 5x5 Matrix   | Protocol 3: LODO Generalization            |
| - 60% Train / 20% Val    | - 20 Transfer Pairs      | - Train: 4 Pooled Domains                  |
| - 20% Held-Out Test      | - 5 Within Diagonals     | - Test: 5th Held-Out Domain                |
| - 5 Seeds (25 runs)      | - 5 Seeds (500 runs)     | - 5 Seeds (100 runs)                       |
+--------------------------+--------------------------+--------------------------------------------+
| Protocol 4: Few-Shot Adaptation (k in {0, 1, 3, 5, 10} support cells, 5,840 runs)               |
+--------------------------------------------------------------------------------------------------+
```

### A. Evaluation Protocol Definitions
1. *Within-Dataset Benchmark:* Evaluated on held-out physical cells within the same repository (60% train, 20% validation, 20% test).
2. *$5 \times 5$ Cross-Dataset Transfer Matrix:* Source-trained model evaluated zero-shot on each target repository ($20$ cross pairs $+ 5$ within diagonals $\times 4$ models $\times 5$ seeds $= 500$ runs).
3. *Leave-One-Dataset-Out (LODO):* Model trained on four pooled datasets and evaluated on the held-out fifth dataset ($5$ target folds $\times 4$ models $\times 5$ seeds $= 100$ runs).
4. *Few-Shot Domain Adaptation:* Target test cells are reserved. Support cells $k \in \{0, 1, 3, 5, 10\}$ are drawn from the remaining target pool across four adaptation strategies (5,840 total runs).

### B. Statistical Replication Across Five Seeds
All experiments were repeated across five independent random seeds:
$$\text{Seeds} \in \{11, 22, 33, 44, 55\}$$
All performance values are reported as $\text{Mean} \pm \text{Standard Deviation}$.

### C. Performance Metrics
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| \quad [\text{SOH percentage points, pp}] \tag{14}$$
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \quad [\text{SOH pp}] \tag{15}$$
$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad [\%] \tag{16}$$

---

## VII. EMPIRICAL BENCHMARK RESULTS

### A. Macro Overview Across All Evaluation Modes
Table II presents the aggregated benchmark scores across all evaluated modes.

```
+--------------------------------------------------------------------------------------------------+
| TABLE II: OVERALL MODEL PERFORMANCE SUMMARY (MEAN ± STD, 5 SEEDS)                                |
+------------------------+------------------+---------------+--------------+-----------------------+
| Evaluation Mode        | Architecture     | Mean MAE (pp) | Std MAE (pp) | Evaluated Runs (N)    |
+------------------------+------------------+---------------+--------------+-----------------------+
| Within-Dataset Holdout | XGBoost          | 3.89          | 3.19         | 25                    |
| Within-Dataset Holdout | Ridge Regression | 4.51          | 3.72         | 25                    |
| Within-Dataset Holdout | ElasticNet       | 4.52          | 3.72         | 25                    |
| Within-Dataset Holdout | Mean Baseline    | 7.87          | 5.06         | 25                    |
| LODO Generalization    | Mean Baseline    | 10.25         | 4.47         | 25                    |
| LODO Generalization    | XGBoost          | 10.82         | 5.44         | 25                    |
| LODO Generalization    | ElasticNet       | 12.69         | 5.80         | 25                    |
| LODO Generalization    | Ridge Regression | 12.72         | 5.84         | 25                    |
| Pairwise Zero-Shot     | Mean Baseline    | 11.22         | 4.24         | 100                   |
| Pairwise Zero-Shot     | XGBoost          | 13.03         | 8.45         | 100                   |
| Pairwise Zero-Shot     | ElasticNet       | 61.31         | 138.63       | 100                   |
| Pairwise Zero-Shot     | Ridge Regression | 61.49         | 138.98       | 100                   |
+------------------------+------------------+---------------+--------------+-----------------------+
```

### B. Within-Dataset Baseline Results
Table III summarizes test performance on held-out physical cells within each repository.

```
+--------------------------------------------------------------------------------------------------+
| TABLE III: WITHIN-DATASET TEST PERFORMANCE ON HELD-OUT CELLS (MEAN ± STD, 5 SEEDS)               |
+----------+-----------+------------------+------------------+------------------+------------------+
| Dataset  | Cathode   | XGBoost MAE (pp) | Ridge MAE (pp)   | ElasticNet (pp)  | Mean Base (pp)   |
+----------+-----------+------------------+------------------+------------------+------------------+
| Oxford   | LCO       | 1.57 ± 0.28      | 1.72 ± 0.50      | 1.73 ± 0.50      | 7.02 ± 1.62      |
| XJTU     | NCM       | 2.18 ± 0.25      | 2.39 ± 0.20      | 2.39 ± 0.20      | 3.84 ± 0.16      |
| MATR     | LFP       | 2.71 ± 0.22      | 2.77 ± 0.20      | 2.78 ± 0.20      | 3.58 ± 0.17      |
| NASA     | LCO       | 4.29 ± 1.39      | 4.41 ± 1.60      | 4.41 ± 1.59      | 9.09 ± 2.09      |
| CALCE    | LCO       | 9.40 ± 2.20      | 11.25 ± 2.06     | 11.27 ± 2.05     | 16.71 ± 1.67     |
+----------+-----------+------------------+------------------+------------------+------------------+
```

### C. Complete $5 \times 5$ Cross-Dataset Transfer Matrix
Table IV presents the complete pairwise transfer matrix. Values represent test MAE in SOH percentage points ($\text{mean} \pm \text{std}$ across 5 seeds). Top entry: **XGBoost**; bottom bracketed entry: **[Ridge Regression]**.

```
+--------------------------------------------------------------------------------------------------+
| TABLE IV: COMPLETE 5x5 CROSS-DATASET PAIRWISE TRANSFER MATRIX (MAE in SOH pp)                    |
| Top: XGBoost | Bottom [Bracketed]: Ridge Regression | Diagonal: Within-Domain Holdout            |
+--------------+------------------+------------------+------------------+------------------+-------+
| Source \ Tgt | CALCE (LCO)      | MATR (LFP)       | NASA (LCO)       | Oxford (LCO)     | XJTU  |
+--------------+------------------+------------------+------------------+------------------+-------+
| CALCE        | 9.40 ± 2.20      | 14.70 ± 4.13     | 10.67 ± 1.44     | 18.93 ± 26.65    | 5.58  |
|              | [11.25 ± 2.06]   | [16.16 ± 3.60]   | [15.05 ± 1.92]   | [133.77 ± 45.32] | [3.87]|
| MATR         | 15.48 ± 0.10     | 2.71 ± 0.22      | 18.18 ± 0.07     | 5.58 ± 0.62      | 3.57  |
|              | [15.72 ± 0.10]   | [2.77 ± 0.20]    | [17.48 ± 0.03]   | [12.91 ± 1.08]   | [3.43]|
| NASA         | 20.42 ± 1.20     | 23.08 ± 2.34     | 4.29 ± 1.39      | 17.10 ± 2.67     | 18.93 |
|              | [68.45 ± 7.95]   | [85.95 ± 8.56]   | [4.41 ± 1.60]    | [637.45 ± 75.52] | [28.6]|
| Oxford       | 17.63 ± 0.15     | 2.92 ± 0.18      | 19.67 ± 0.28     | 1.57 ± 0.28      | 4.59  |
|              | [16.67 ± 0.12]   | [2.73 ± 0.05]    | [17.03 ± 0.23]   | [1.72 ± 0.50]    | [3.58]|
| XJTU         | 13.06 ± 0.07     | 8.01 ± 0.13      | 17.34 ± 0.08     | 6.13 ± 0.44      | 2.18  |
|              | [11.51 ± 0.04]   | [11.70 ± 0.36]   | [17.15 ± 0.27]   | [110.49 ± 4.65]  | [2.39]|
+--------------+------------------+------------------+------------------+------------------+-------+
```

### D. Leave-One-Dataset-Out (LODO) Generalization Results
Table V reports LODO generalization performance across all five held-out domains.

```
+--------------------------------------------------------------------------------------------------+
| TABLE V: LEAVE-ONE-DATASET-OUT (LODO) GENERALIZATION PERFORMANCE (MEAN ± STD, 5 SEEDS)           |
+-----------------+-------------+------------------+------------------+------------------+---------+
| Held-Out Domain | Target Chem | XGBoost MAE (pp) | Ridge MAE (pp)   | ElasticNet (pp)  | Mean Base|
+-----------------+-------------+------------------+------------------+------------------+---------+
| XJTU            | NCM         | 3.60 ± 0.08      | 3.29 ± 0.06      | 3.30 ± 0.06      | 4.17    |
| Oxford          | LCO         | 5.88 ± 0.67      | 19.08 ± 1.34     | 18.91 ± 1.34     | 7.69    |
| MATR            | LFP         | 11.92 ± 0.75     | 9.40 ± 1.25      | 9.39 ± 1.25      | 9.13    |
| CALCE           | LCO         | 15.38 ± 0.15     | 15.69 ± 0.11     | 15.69 ± 0.11     | 16.29   |
| NASA            | LCO         | 17.34 ± 0.20     | 16.16 ± 0.36     | 16.14 ± 0.36     | 13.95   |
+-----------------+-------------+------------------+------------------+------------------+---------+
```

### E. Few-Shot Domain Adaptation Progression
Table VI reports few-shot adaptation performance across support sizes $k \in \{0, 1, 3, 5, 10\}$.

```
+--------------------------------------------------------------------------------------------------+
| TABLE VI: FEW-SHOT TARGET DOMAIN ADAPTATION PROGRESSION (MAE in SOH pp)                          |
+------------------+---------------+-----------+-----------+-----------+-----------+---------------+
| Architecture     | Strategy      | k = 0     | k = 1     | k = 3     | k = 5     | k = 10        |
+------------------+---------------+-----------+-----------+-----------+-----------+---------------+
| XGBoost          | none (zero)   | 13.08 pp  | —         | —         | —         | —             |
| XGBoost          | scratch       | —         | 7.75 pp   | 5.00 pp   | 4.16 pp   | 4.98 pp       |
| XGBoost          | pooled        | —         | 9.43 pp   | 6.45 pp   | 5.62 pp   | 5.96 pp       |
| XGBoost          | coral         | —         | 11.82 pp  | 10.26 pp  | 10.18 pp  | 10.82 pp      |
| XGBoost          | normalization | —         | 11.82 pp  | 10.26 pp  | 10.17 pp  | 10.82 pp      |
| Ridge Regression | none (zero)   | 62.88 pp  | —         | —         | —         | —             |
| Ridge Regression | scratch       | —         | 14.34 pp  | 5.33 pp   | 4.92 pp   | 5.69 pp       |
| Ridge Regression | pooled        | —         | 15.04 pp  | 8.70 pp   | 7.73 pp   | 6.98 pp       |
| Ridge Regression | coral         | —         | 19.03 pp  | 10.45 pp  | 10.15 pp  | 11.00 pp      |
| Ridge Regression | normalization | —         | 19.07 pp  | 10.45 pp  | 10.15 pp  | 11.00 pp      |
| Overall Macro    | scratch       | —         | 11.18 pp  | 5.80 pp   | 5.40 pp   | 6.09 pp       |
| Overall Macro    | pooled        | —         | 12.44 pp  | 8.19 pp   | 7.44 pp   | 7.14 pp       |
+------------------+---------------+-----------+-----------+-----------+-----------+---------------+
```

### F. Accounting of Physically Infeasible Settings
To uphold empirical integrity, when requested support size $k$ exceeded the available cells after reserving test cells, runs were explicitly logged as infeasible, summarized in Table VII.

```
+--------------------------------------------------------------------------------------------------+
| TABLE VII: ACCOUNTING OF PHYSICALLY INFEASIBLE ADAPTATION SETTINGS                               |
+---------------+-------------+-----------+----------------------+---------------------------------+
| Target Domain | Total Cells | Test Held | Requested Support k  | Explicit Physical Reason        |
+---------------+-------------+-----------+----------------------+---------------------------------+
| NASA          | 4 cells     | 2 cells   | k in {3, 5, 10}      | Only 2 support cells available  |
| Oxford        | 8 cells     | 3 cells   | k = 10               | Only 5 support cells available  |
| Total Runs    | —           | —         | 320 logged runs      | Exactly logged in failure CSV   |
+---------------+-------------+-----------+----------------------+---------------------------------+
```

---

## VIII. ELECTROCHEMICAL DOMAIN SHIFT AND DIAGNOSTIC FAILURE ANALYSIS

### A. The Linear Extrapolation Pathology
A critical finding of this benchmark is the dramatic divergence of unconstrained linear models (Ridge and ElasticNet) under cycle-lifespan domain shifts.
- In NASA, battery aging is recorded over a short lifespan (132 to 168 cycles). Fitting a linear model on NASA yields a steep degradation slope ($w \approx -0.15\%$ SOH/cycle).
- When evaluated zero-shot on Oxford (cycles extend to 520), the unconstrained linear model extrapolates downwards into deeply negative, unphysical SOH values:
  $$\hat{y}(500) = w \cdot 500 + y_0 \approx (-0.15)(500) + 100 \approx 25\% \text{ or negative}$$
  Producing a catastrophic MAE of $637.45 \pm 75.52$ pp.
- In contrast, tree-based models (XGBoost) partition the input cycle space using orthogonal threshold splits. For test cycles exceeding the maximum training split ($x > x_{\max,\text{train}}$), decision trees predict the constant leaf value of the highest observed split. Consequently, XGBoost error remains strictly bounded at $17.10 \pm 2.67$ pp.

### B. The Accidental Alignment Phenomenon: Oxford $\to$ MATR
Despite radical electrochemical disparities:
- *Chemistry:* Oxford is LCO pouch; MATR is LFP cylindrical.
- *Charging Protocol:* Oxford used 0.5C standard charge; MATR used 3.6C–6.0C ultra-fast charging.
- *Ambient Temperature:* Oxford was tested at 40°C; MATR at 30°C.
Zero-shot transfer from Oxford $\to$ MATR achieved an exceptional MAE of **$2.73 \pm 0.05$ pp for Ridge** and **$2.92 \pm 0.18$ pp for XGBoost**—matching the accuracy of training directly on MATR held-out cells ($2.71$ pp). This occurs because Kokam pouch LCO cells cycled at 40°C under dynamic Artemis driving experience accelerated linearized capacity loss that electrochemically mirrors the linearized degradation rate of A123 LFP cells under high C-rate cycling.

---

## IX. PRACTICAL BMS IMPLICATIONS AND LIMITATIONS

### A. Recommendations for On-Board Automotive BMS
1. *Constrain Linear Estimators with Physical Saturation:* Linear models should never be deployed across cell formats without saturation clamps ($[0\%, 105\%]$ SOH) or cycle boundary thresholds.
2. *Few-Shot Calibration is Highly Practical:* Automotive manufacturers can achieve pack-level SOH accuracy by collecting calibration data from as few as 1 to 3 support cells ($k = 1$ reduces error by $>60\%$).
3. *Multi-Domain Pooling as an Industrial Baseline:* Production BMS models should be pre-trained on multi-domain pooled datasets (as demonstrated in LODO) to eliminate single-chemistry vulnerability.

### B. Limitations and Threats to Validity
- *Legacy Benchmark Sample Sizes:* Small cell counts in NASA (4 cells) and Oxford (8 cells) increase variance across random seeds.
- *Single-Feature Baseline:* The conservative baseline evaluated is cycle age. Future studies should integrate partial charging curves ($dQ/dV$) once synchronized across repositories.

---

## X. CONCLUSION AND FUTURE DIRECTIONS

This work establishes one of the largest and most methodologically rigorous empirical benchmarks for cross-dataset EV battery SOH generalization. Evaluating 190 physical cells across 5 public repositories over 6,440 benchmark runs, we demonstrated that within-domain laboratory accuracy does not guarantee transferability. Unconstrained linear models suffer catastrophic extrapolation failure under cycle-span mismatches, whereas tree-based ensembles maintain bounded error. Multi-domain pooling (LODO) and few-shot adaptation ($k = 1$ to $3$) effectively bridge the domain gap, providing a clear roadmap for robust, real-world EV battery management systems.

---

## ACKNOWLEDGMENT
The author thanks the faculty advisors and laboratory coordinators of the Department of Computer Science and Engineering, BMS College of Engineering, Bangalore, for guidance during BMSCE Project Work 2. The author also acknowledges the open-science contributions of NASA Ames PCoE, CALCE, XJTU, Oxford University, and the Severson et al. research group for public data dissemination.

---

## REFERENCES

[1] K. A. Severson, P. M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M. H. Chen, M. Aykol, P. K. Herring, D. Fraggedakis, and others, "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy*, vol. 4, no. 5, pp. 383–391, May 2019.

[2] B. Saha and K. Goebel, "Battery data set," *NASA Ames Prognostics Data Repository*, NASA Ames Research Center, Moffett Field, CA, 2007.

[3] M. Pecht, "CALCE battery data," *Center for Advanced Life Cycle Engineering (CALCE)*, University of Maryland, College Park, MD, 2011.

[4] C. R. Birkl, "Oxford battery degradation dataset," *Department of Engineering Science*, University of Oxford, Oxford, U.K., 2017.

[5] H. Sun, Y. He, and F. Sun, "A large-scale lithium-ion battery dataset for state of health estimation and remaining useful life prediction," *Xi'an Jiaotong University Dataset Repository*, Xi'an, China, 2022.

[6] J. Tian, R. Xiong, and W. X. Shen, "State-of-health estimation based on differential charging voltage for lithium-ion batteries," *IEEE Transactions on Power Electronics*, vol. 35, no. 10, pp. 10380–10391, Oct. 2020.

[7] M. A. Hannan, M. S. H. Lipu, A. Hussain, and A. Mohamed, "A review of lithium-ion battery state detection using artificial intelligence," *IEEE Access*, vol. 7, pp. 31781–31804, Mar. 2019.

[8] X. Hu, S. Li, and H. Peng, "A comparative study of equivalent circuit models for Li-ion batteries," *Journal of Power Sources*, vol. 198, pp. 359–367, Jan. 2012.

[9] B. Sun and K. Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation," in *Proc. European Conference on Computer Vision (ECCV)*, Amsterdam, Netherlands: Springer, Oct. 2016, pp. 443–450.
