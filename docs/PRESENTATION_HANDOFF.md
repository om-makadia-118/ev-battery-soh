# Presentation Brief — EV Battery State-of-Health Research

**Project title:** Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction  
**Presentation context:** BMSCE Project Work 2  
**Status date:** 30 October 2026

## 1. What is the project about?

Our project investigates whether machine-learning models trained using one lithium-ion battery dataset can accurately estimate battery State of Health (SOH) on batteries from completely different datasets.

SOH measures how much capacity a battery retains compared with its rated capacity:

\[
SOH(\%) = \frac{\text{Measured discharge capacity}}{\text{Rated capacity}} \times 100
\]

The main challenge is that battery datasets differ in chemistry, nominal capacity, temperature, charging protocol, measurement equipment and data format.

A model that performs well on one dataset may not generalize to another. Our project aims to systematically investigate this problem through a common data-processing and evaluation framework.

## 2. Work completed so far

The strongest way to present our progress is as the development of the research and data-engineering foundation.

### A. Research design and dataset selection

We identified five public battery datasets covering different battery chemistries and testing conditions:

- NASA.
- CALCE.
- XJTU.
- Oxford.
- MIT-Stanford-Toyota (MATR).

We also studied the important dataset characteristics, including nominal capacities, available cells, measurement formats and EOL definitions.

This gave us a multi-dataset research scope instead of limiting the work to a single battery dataset.

### B. Dataset acquisition and investigation

We collected or identified source files across several different formats:

- MATLAB files for NASA, Oxford and MATR.
- Pickle files for XJTU.
- Excel workbooks for CALCE Arbin records.
- Text-based CADEX files for CALCE.

We investigated the actual structure of representative files instead of assuming that all datasets followed the same schema.

Some important corrections were made during this process:

- The currently identified NASA local scope is four batteries: B0005, B0006, B0007 and B0018. This is different from the 28-cell scope listed in the initial review.
- The currently available XJTU files are 15 cells from Batch 2 under the 3C condition, not the full 55-cell dataset.
- CALCE work is focused on the CS2 cycling dataset.
- Oxford contains sparse capacity-characterization records, so its labels must not be artificially interpolated.

### C. Dataset-specific parsing investigation

One major engineering challenge is that every dataset stores its information differently.

We investigated the different source formats and developed/maintained readers and parsing logic in the project framework.

A particularly important example is CALCE CADEX:

- Charging and discharging records are identified using status codes.
- Voltage and current require unit conversion.
- Time is represented as duration in seconds.
- The `Capacity` column is a progress percentage, not a capacity in Ah.
- Actual discharge capacity is obtained by integrating current over time.

For one inspected CS2_8 file, 11 discharge cycles were extracted, with capacities approximately in the 1.10–1.12 Ah range.

This illustrates why source-specific data investigation is necessary before calculating SOH.

### D. Common data-processing framework

We developed a common data-processing direction so that different datasets can ultimately be represented consistently.

The intended processing pipeline is:

**Raw files → Dataset-specific readers → Unit normalization → Cycle/cell mapping → Capacity extraction → SOH calculation → Data-quality validation → Unified dataset**

The common SOH formula is:

\[
SOH(\%) = \frac{Capacity_{Ah}}{RatedCapacity_{Ah}} \times 100
\]

This allows different datasets to be compared using a consistent target definition while preserving source metadata.

### E. Research framework and evaluation design

The v4 project contains a research framework for comparing classical and deep-learning approaches.

The documented model families include:

**Classical models**
- Mean baseline.
- Ridge Regression.
- ElasticNet.
- XGBoost.

**Deep-learning models**
- LSTM.
- 1D-CNN.
- Transformer.

The framework also describes within-dataset, cross-dataset and leave-one-dataset-out evaluation, along with few-shot adaptation and CORAL.

These are framework capabilities, not claims that the complete real-data benchmark has already been executed.

### F. Research integrity and validation design

We have identified several important safeguards:

- Split training and testing data by complete battery IDs.
- Do not put cycles from the same battery in both training and test sets.
- Do not use the current cycle's measured capacity or SOH as a predictor.
- Preserve source EOL definitions.
- Keep Oxford's sparse observations as they are.
- Report missing or infeasible experiment settings honestly.
- Compare against a naive mean baseline.
- Use multiple random seeds for final benchmark reporting.

These safeguards are important because an apparently accurate model can otherwise benefit from data leakage or inconsistent preprocessing.

## 3. Challenges encountered and how we addressed them

| Challenge | Why it matters | Engineering response |
|---|---|---|
| Multiple data formats | MATLAB, pickle, Excel and text files cannot be processed identically | Investigated source structures and used dataset-specific readers/mappings |
| Different capacity units | Incorrect units directly corrupt SOH labels | Explicit unit conversion and capacity-semantic checks |
| CADEX capacity field ambiguity | The field is a progress percentage, not Ah | Calculate discharge capacity from current and elapsed time |
| Different battery counts | Initial report scope differs from available local files | Corrected the inventory and distinguished available files from full dataset scope |
| Sparse Oxford measurements | Interpolation creates unobserved labels | Preserve only measured characterization records |
| Different EOL definitions | Comparisons can become inconsistent | Preserve source-specific thresholds and document common-threshold analysis separately |
| Data leakage risk | Random cycle splitting can inflate accuracy | Use battery-level splitting |
| Large MATR files | Cell identity and batch mapping need care | Keep identity verification as an explicit validation task |

## 4. Current project status

The project has progressed beyond the initial research proposal. The research scope, source investigation, common preprocessing direction, software framework and evaluation safeguards have been established.

| Workstream | Current status |
|---|---|
| Research problem and objectives | Defined |
| Five-dataset research scope | Defined |
| Dataset acquisition and inventory | Substantial progress; actual coverage needs final verification |
| Source-format investigation | Completed for representative files across multiple formats |
| Dataset reader/preprocessing framework | Implemented in the project framework; full real-source validation pending |
| SOH calculation methodology | Defined and implemented in the framework; source-by-source QC pending |
| Data-quality and leakage safeguards | Designed and represented in the framework |
| Classical/deep-learning experiment framework | Present in the codebase; real-data results pending |
| Streamlit research dashboard | Presentation interface being added |
| Full real-data benchmark | Pending |
| Final result analysis and paper | Pending |

### Key status message

The major groundwork is in place. The remaining research phase is concentrated on final source-data validation, creation of the verified unified dataset, and then running and analyzing the benchmark experiments.

Do not say that the five datasets have all been completely processed or that model accuracy results are already available.

## 5. Suggested presentation structure

### Slide 1 — Title and objective

**Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction**

Introduce the research question:

Can an SOH prediction model trained on one battery dataset generalize to another battery dataset?

### Slide 2 — Problem statement and motivation

Explain:

- Battery degradation reduces usable capacity.
- SOH estimation is important for EV battery assessment.
- Different datasets have different chemistries and operating conditions.
- Models may fail when applied outside their training dataset.

### Slide 3 — Datasets investigated

Show the five datasets:

- NASA — LCO.
- CALCE — LCO.
- XJTU — NCM.
- Oxford — LCO.
- MATR — LFP.

Mention nominal capacities and operating conditions where relevant.

Important: distinguish the intended dataset size from the files currently available. The current identified local scope is four NASA batteries and 15 XJTU batteries.

### Slide 4 — Data engineering challenges

Show the variety of input formats:

- MATLAB.
- Pickle.
- Excel.
- CADEX text.

Explain the unit and capacity-semantic problems using CALCE CADEX as a concrete example.

### Slide 5 — Proposed preprocessing architecture

Use this flow:

Raw datasets → Dataset-specific parsing → Unit normalization → Common schema → SOH calculation → Data-quality checks → Unified dataset.

Explain why this step is necessary before model comparisons.

### Slide 6 — Research and evaluation framework

Show the model families already represented in the codebase:

- Mean, Ridge, ElasticNet and XGBoost.
- LSTM, 1D-CNN and Transformer.

Then explain the planned evaluation:

- Within-dataset.
- Cross-dataset.
- Leave-one-dataset-out.
- Few-shot adaptation.
- CORAL alignment.

Clearly state that these are the planned/implemented framework capabilities and that the complete real-data benchmark is still pending.

### Slide 7 — Current implementation and live demonstration

Use the Streamlit dashboard to show:

1. Project overview.
2. Dataset inventory.
3. Data-processing workflow.
4. SOH calculation.
5. Validation safeguards.
6. Planned experiment matrix.

If a verified unified dataset is not available, show the workflow and dataset registry rather than presenting synthetic values as experimental output.

### Slide 8 — Challenges, status and next steps

Summarize the challenges addressed and the remaining work:

- Finalize and validate source parsers.
- Re-run preparation using v4.
- Generate the verified unified dataset.
- Run the experiment matrix.
- Analyze transfer performance and failures.
- Prepare final research results.

End with the objective: identify when battery SOH models generalize and when domain adaptation is necessary.

## 6. Suggested live demo script

A simple demonstration can take 3–4 minutes.

**Step 1 — Project overview**

"This dashboard summarizes the research scope, the datasets investigated and the current implementation status."

**Step 2 — Dataset registry**

"These datasets come from different sources and use different formats, so we first investigated the source structures and the correct capacity measurements."

**Step 3 — Data pipeline**

"We convert the different raw formats into a common representation. We calculate SOH from measured discharge capacity and rated capacity."

**Step 4 — Validation**

"We also enforce battery-level separation and prevent current-cycle capacity leakage. This is important for obtaining meaningful generalization results."

**Step 5 — Research plan**

"Once the unified data is validated, the existing experiment framework will compare classical and deep-learning models under within-dataset and cross-dataset settings."

## 7. Likely questions and suggested answers

**Q1. Why are you using multiple datasets?**

Because a model trained and tested on the same dataset may not generalize to different battery chemistries, testing conditions or measurement systems. Our objective is to study that transfer problem.

**Q2. Why not simply combine all datasets and train one model?**

The datasets have different source formats, units, battery characteristics and measurement protocols. They need careful normalization first. Also, combining them without a proper battery-level and dataset-level evaluation could hide generalization failures.

**Q3. How do you calculate SOH?**

SOH is measured discharge capacity divided by rated capacity, multiplied by 100.

**Q4. How do you avoid data leakage?**

All observations belonging to one battery are kept in the same split. Current-cycle measured capacity and SOH are not used as predictors of current-cycle SOH.

**Q5. Why is the complete benchmark not presented yet?**

We are prioritizing data correctness. Incorrect capacity semantics, unit conversions or battery mapping would make model comparisons unreliable. The next phase is to validate the unified dataset and then run the benchmark.

**Q6. Have the models been trained?**

The repository contains the model and experiment framework, but complete real-data benchmark results are not yet available. We will report them after the source data and evaluation protocol are validated.

**Q7. Why is the Oxford dataset different?**

Oxford has sparse capacity-characterization checkpoints. We retain measured observations rather than creating artificial labels through interpolation.

## 8. Important presentation wording

Prefer:

- "We developed the common data-processing framework."
- "We investigated the source-specific data formats."
- "We established the SOH calculation and validation methodology."
- "The experiment framework is available in the codebase."
- "The next phase is real-data validation and benchmarking."

Avoid:

- "All five datasets are fully processed."
- "We achieved a particular model accuracy" unless supported by actual results.
- "The model successfully generalizes across all chemistries."
- "The full 28 NASA and 55 XJTU cells are available locally."
- "The complete benchmark is finished."

**Overall message:** The project has established a substantial research and engineering foundation. The remaining work is focused on finalizing real-data validation and producing the empirical evidence needed to answer the cross-dataset generalization question.