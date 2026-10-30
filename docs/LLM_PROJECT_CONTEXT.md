# EV Battery SOH Research Project — Complete LLM Handoff

**Project:** Cross-Dataset Generalization of Machine-Learning Models for EV Battery State-of-Health Prediction  
**Institutional context:** BMSCE Project Work 2  
**Context last updated:** 30 October 2026  
**Purpose:** Technical handoff for another LLM, researcher or developer continuing this project.

---

# 1. Instructions for the next LLM

Read this document before modifying the repository.

The most important instructions are:

1. **Use the v4 project ZIP as the code baseline.** Do not replace it with the older v2 code.
2. The uploaded `ev_battery_soh_handoff.zip` was created using v2. It is a historical handoff containing potentially reusable raw data, inventory information, logs and older code. Its processed outputs must not automatically be trusted.
3. Do not start implementing additional models or training models until the user explicitly asks. The current priority is to make the existing project, data pipeline, documentation and presentation interface proper.
4. Do not fabricate dataset coverage, successful parser validation, test results, benchmark metrics or research conclusions.
5. Do not use synthetic data as real experimental results.
6. Do not randomly split cycle rows. All evaluation splits must be performed at the battery/cell level.
7. Never use the current cycle's measured capacity or SOH as a model input when predicting that same cycle's SOH.
8. Preserve the original dataset metadata, units, battery identifiers and source EOL definitions.
9. Do not interpolate unobserved SOH labels, particularly for Oxford's sparse characterization records.
10. Report failures and unsupported configurations instead of silently manufacturing successful results.

The user prefers practical, complete implementation work and does not want repetitive requests to manually perform one small task at a time.

# 2. Project objective

The research investigates:

**How well do machine-learning models trained on one lithium-ion battery dataset generalize to batteries from other datasets, chemistries, operating conditions and testing protocols?**

The project is not simply an SOH prediction exercise. Its central research focus is cross-dataset generalization.

A model can perform well when training and testing on cells from the same dataset but may fail when applied to another dataset because of differences in:

- Battery chemistry and cell characteristics.
- Nominal capacity.
- Charging and discharging protocols.
- Temperature.
- Measurement equipment and sampling frequency.
- Cycle definitions and capacity characterization frequency.
- Data formats and measurement units.

The intended contribution is a consistent benchmark and analysis of these domain differences, rather than claiming to invent an entirely new neural-network architecture.

# 3. What SOH means

State of Health (SOH) represents the remaining capacity of a battery relative to its rated or nominal capacity.

The project uses:

\[
SOH(\%) =
\frac{\text{Measured discharge capacity (Ah)}}
{\text{Rated capacity (Ah)}} \times 100
\]

Important rules:

- Use measured discharge capacity for the corresponding characterization record.
- Use the appropriate rated capacity for the battery/source.
- Preserve values above 100% when measurements justify them.
- Do not clip SOH to 100%.
- Do not use measured capacity or SOH from the current prediction target as a model feature.
- Keep the source EOL definition separate from any common harmonized EOL analysis.

# 4. Dataset scope and known inventory

The research plan covers five public battery datasets.

The figures below are the currently reported/identified inventory from earlier project work. They are **not a declaration that all five datasets have been fully validated by the v4 pipeline**.

| Dataset | Chemistry | Nominal capacity | Scope in original review | Currently identified/downloaded scope | Important caveat |
|---|---|---:|---:|---:|---|
| NASA | LCO | 2 Ah | 28 batteries | 4 batteries | Local files identified are B0005, B0006, B0007 and B0018 |
| CALCE | LCO | 1.1 Ah | 15 batteries | 15 CS2 cells | Arbin Excel and CADEX text formats require consistent validation |
| XJTU | NCM | 2 Ah | 55 batteries | 15 batteries | The available files are Batch 2, 3C; not the complete 55-cell collection |
| Oxford | LCO | 0.74 Ah | 8 batteries | 8 cells | Characterization measurements are sparse, approximately every 100 cycles |
| MIT-Stanford-Toyota (MATR) | LFP | 1.1 Ah | 124 batteries | Original 124-cell collection targeted | Three batch MAT files; cross-batch cell identity mapping must be verified |

## 4.1 NASA

Known files:

- B0005.mat
- B0006.mat
- B0007.mat
- B0018.mat

The locally identified scope is four batteries, not the 28 listed in the original review's dataset table.

The project includes MATLAB data-reading support. The exact source files must still be run through the final v4 preparation and quality checks.

The original review lists a 70% source EOL threshold.

## 4.2 CALCE

The intended source is the CALCE CS2 cycling dataset, not the separate CALCE storage experiment.

The user has downloaded 15 CS2 archives.

Two formats are relevant:

- Arbin Excel workbooks.
- CADEX text files.

Files previously inspected include:

- `CS2_8_1_19_10.txt`
- `CS2_33_1_10_11.xlsx`

### CADEX parsing observations

The previously inspected CADEX file showed:

- Status category 3 represents charging.
- Status category 6 represents discharging.
- Voltage values are recorded in millivolts and must be converted to volts.
- Current values are recorded in milliamps and must be converted to amps.
- Duration is in seconds.
- The `Capacity` column represents progress percentage, not Ah.
- Discharge capacity must be calculated by integrating discharge current over elapsed time.

For the previously inspected CS2_8 file, 11 discharge cycles were extracted, with approximate capacities around 1.10–1.12 Ah.

These findings are useful parser evidence, but they do not establish that every CALCE file has been successfully parsed and validated.

The original review lists a 70% source EOL threshold.

## 4.3 XJTU

The currently available files are 15 pickle files from Batch 2 under the 3C operating condition.

One inspected battery pickle contained 298 cycles.

The inspected structure included fields for:

- Cycle number.
- Current.
- Voltage.
- Charge capacity.
- Discharge capacity.
- Time.
- Temperature.
- Internal resistance.

The available 15 files must not be described as the full 55-cell XJTU collection.

The original review lists an 80% source EOL threshold.

## 4.4 Oxford

The dataset contains eight cells.

Known characteristics:

- LCO cells.
- Approximately 740 mAh nominal capacity.
- 40°C testing condition.
- MATLAB `.mat` files.
- Sparse capacity-characterization checkpoints, approximately every 100 cycles.

Important: do not interpolate SOH labels between Oxford characterization checkpoints merely to create a denser time series. The observation frequency is a property of the source dataset.

The original review lists an EOL threshold of approximately 80%.

## 4.5 MATR / MIT-Stanford-Toyota

The intended dataset is the original 124-cell LFP collection associated with the Severson et al. work.

Known source structure:

- Three original batch MAT files.
- Approximately 1.1 Ah nominal cell capacity.
- Large source files, potentially several gigabytes each.

A summary reader exists in the project history, but cell identity and batch mapping must be verified before treating the combined collection as a validated 124-cell dataset.

The original review lists an 80% source EOL threshold.

# 5. EOL policy: important design decision

The original project review lists different source EOL definitions:

| Dataset | Source EOL |
|---|---:|
| NASA | 70% |
| CALCE | 70% |
| XJTU | 80% |
| Oxford | Approximately 80% |
| MATR | 80% |

Some earlier project documentation also describes an 80% common threshold.

These definitions must not be silently conflated.

Recommended handling:

1. Preserve the source-specific EOL threshold and its provenance.
2. Add a separate harmonized 80% EOL analysis where scientifically appropriate.
3. Clearly identify which threshold is used in every experiment.
4. Do not compare EOL-based outcomes if different rules have been applied without documenting them.

# 6. Technical architecture

The project is a Python research pipeline.

The v4 README describes the following architecture and capabilities:

## Data layer

Responsibilities:

- Read dataset-specific source files.
- Convert source observations into a canonical representation.
- Normalize units.
- Associate observations with battery and dataset identifiers.
- Calculate SOH.
- Check missing values, invalid capacities and metadata.
- Generate data-quality reports.

Relevant project areas:

- `battery_soh/data/`
- Dataset readers and schema.
- SOH and feature handling.
- Battery-level splits.

## Model layer

The existing v4 framework describes:

### Classical methods

- Mean baseline.
- Ridge Regression.
- ElasticNet.
- XGBoost.

### Deep-learning methods

- LSTM.
- 1D-CNN.
- Transformer.

The presence of these implementations in the repository must not be confused with successful training on validated real data.

**Current instruction: do not add new model implementations or start training.** First finalize the existing project and data foundation.

## Experiment layer

The framework describes:

- Within-dataset evaluation.
- Cross-dataset evaluation.
- Leave-one-dataset-out evaluation.
- Few-shot adaptation.
- CORAL-based domain alignment.

Relevant project area:

- `battery_soh/experiments/`

## Evaluation layer

The framework describes:

- MAE.
- RMSE.
- MAPE.
- Prediction exports.
- Failure reports.
- Transfer matrices.
- Model ranking plots.
- Per-pair/per-seed summaries.
- Error analysis by chemistry, temperature and charging speed where metadata supports it.

Relevant project area:

- `battery_soh/evaluation/`

# 7. Canonical data representation

The project should maintain one common representation across all datasets.

The exact final column names must be reconciled with the v4 implementation rather than independently invented by a new LLM.

Conceptually, the canonical record should retain:

| Field | Meaning |
|---|---|
| dataset | Source dataset identifier |
| battery_id | Unique cell/battery identifier within the source |
| cycle_index / cycle_number | Original or normalized cycle position |
| capacity_ah | Measured discharge capacity for that observation |
| rated_capacity_ah | Rated capacity used for SOH normalization |
| soh_percent | Calculated SOH target |
| temperature_c | Temperature where available |
| charge_rate_c | Charging rate where available |
| discharge_rate_c | Discharging rate where available |
| source_file | Original source filename or provenance |
| source_cycle_id | Original cycle/record reference where available |

Do not invent missing metadata.

A value that is unavailable in the source should remain missing or explicitly marked unavailable, rather than being silently inferred.

# 8. Data preparation and integrity requirements

The preparation process should be:

1. Inventory the raw files.
2. Identify each source format and metadata structure.
3. Inspect representative files from each source.
4. Validate source units.
5. Extract measured discharge capacity.
6. Map the correct battery identifier.
7. Map cycle or characterization record identifiers.
8. Calculate SOH from rated capacity.
9. Preserve source metadata and EOL definitions.
10. Validate duplicates, missing values, invalid records and capacity ranges.
11. Produce the unified dataset.
12. Produce a data-quality report.
13. Verify the resulting dataset before running experiments.

Never silently discard parsing failures.

A parser should either produce validated records or report a meaningful failure with the affected source file and reason.

# 9. Feature and leakage policy

The conservative common feature currently described by the project README is **cycle number only**.

This is an age-based baseline, not a validated charging-curve-only estimator.

The long-term research objective may use common physical features from charging/discharging data, but only after verifying that they are genuinely available and comparable across the datasets.

Prohibited current-cycle target leakage:

- Current-cycle measured discharge capacity.
- Current-cycle SOH.
- Any feature calculated directly from the target value.

Sequence models must not consume future target values.

# 10. Evaluation protocol

The intended research evaluation has several levels.

## 10.1 Within-dataset

Train and test on different batteries from the same dataset.

All cycles belonging to a battery must stay in the same split.

## 10.2 Cross-dataset

Train on one dataset and test on another dataset whose batteries were not used during training.

The intended output is a 5×5 transfer matrix across the five datasets.

## 10.3 Leave-one-dataset-out

Train on four datasets and evaluate on the fifth dataset.

This tests generalization to an unseen dataset domain.

## 10.4 Few-shot adaptation

Use a small number of target-domain support batteries to adapt the source model.

The planned support settings are:

- 1 battery.
- 3 batteries.
- 5 batteries.
- 10 batteries.

Support batteries must be disjoint from target test batteries.

If a dataset does not contain enough batteries for a requested setting, report it as infeasible. Never manufacture a result.

## 10.5 CORAL

CORAL is intended to align source and target feature distributions.

The alignment must only use permitted training-source and target-support information. It must not use target test labels or target test data to fit transformations.

## 10.6 Random seeds and baselines

The original plan calls for five random seeds and reporting mean plus spread.

The naive mean predictor is a required baseline.

Every model must be compared against the same evaluation protocol.

# 11. Current project status

This is the working status based on the project materials and earlier inspection history. It must be refreshed after the actual v4 repository and raw files are run together.

| Area | Status | Interpretation |
|---|---|---|
| Research question and scope | Defined | Cross-dataset generalization is the main objective |
| Dataset selection | Defined | Five public sources selected |
| Dataset acquisition | Partial | Several source files collected; counts differ from original review scope |
| Source-format investigation | Substantial groundwork | MATLAB, pickle, Excel and CADEX structures investigated |
| Canonical schema | Implemented in framework | Must be verified against every actual source |
| SOH calculation | Implemented in framework | Must be validated against actual source capacity semantics |
| Dataset readers | Present in v4 framework | Actual-file coverage and edge cases require verification |
| Data-quality checks | Present in framework | Must be run on the final raw inventory |
| Battery-level split logic | Present in framework | Must be retained and audited |
| Classical model framework | Present in v4 | Real benchmark training not yet claimed |
| Deep-learning framework | Present in v4 | Real benchmark training not yet claimed |
| Cross-dataset experiment framework | Present in v4 | Real benchmark results pending |
| Few-shot/CORAL framework | Present in v4 | Real benchmark results pending |
| Real unified dataset | Not yet certified | Requires final v4 preparation and QC |
| Real benchmark metrics | Not available | Do not fabricate |
| Presentation dashboard | To be included in this update | Must clearly label real data vs illustrative data |

The earlier v4 checkpoint was reported to have 31 passing tests. These were synthetic unit/integration tests, not proof of complete real-source validation. Re-run the current test suite before making a fresh test-pass claim.

# 12. Version and artifact provenance

There are two important archives:

## v4 project archive

This is the authoritative code baseline.

Use it for:

- Project source code.
- Current architecture.
- Current readers and parsers.
- Existing model/experiment framework.
- Tests and documentation.

## v2 handoff archive

Filename: `ev_battery_soh_handoff.zip`

Size: approximately 376.6 MB.

The archive was generated using v2 code.

It was intended to contain:

- Project source and tests.
- Dataset inventory.
- Inspection logs.
- Data preparation logs.
- Test logs.
- A limited selection of raw source files, with a collection size cap.

Use it as a historical source of evidence and potentially reusable raw data.

Do not:

- Replace v4 with the v2 source.
- Treat v2 processed CSVs as final.
- Treat v2 parser results as v4 validation.
- Assume every expected raw file was included.
- Claim the archive's contents have been fully audited until extraction and inventory are successful.

Any raw files recovered from v2 should be reprocessed through the v4 pipeline.

# 13. Challenges encountered and engineering lessons

## Heterogeneous formats

The datasets use MATLAB, pickle, Excel and text formats.

A single generic CSV loader is insufficient for all of them. Dataset-specific readers and mappings are needed.

## Capacity semantics

A field called `Capacity` does not necessarily mean measured Ah. The CALCE CADEX file demonstrated this: its capacity field represented progress percentage.

Always verify units and semantics before calculating SOH.

## Unit conversion

Some source readings use mV/mA while the canonical representation uses V/A. Duration may be in seconds.

Conversions must be explicit and tested.

## Battery-count discrepancy

The original review lists 28 NASA and 55 XJTU batteries, but the currently identified local scope is four NASA and 15 XJTU cells.

The report and presentation must distinguish target dataset scope from locally available data.

## Sparse characterization

Oxford's measurements are not available for every physical cycle. Interpolation would create artificial labels and distort the evaluation.

## Battery-level leakage

Random cycle-level splitting can place cycles from the same battery in both training and test sets. This can make reported performance misleading.

Split by whole battery identifiers.

## EOL inconsistency

The original sources use different EOL definitions. Preserve source rules and document any harmonized threshold separately.

## MATR identity mapping

Large batch files require careful verification of cell identity and batch membership before combining records.

# 14. Current scope freeze

Until the user explicitly changes this instruction:

- Do not implement additional ML/DL models.
- Do not start model training.
- Do not produce fabricated benchmark charts.
- Do not add unnecessary architecture or infrastructure.
- Focus on codebase cleanup, documentation, data preparation correctness, project reproducibility and presentation readiness.

# 15. Recommended next steps

Execute in this order:

1. Inspect the v4 repository structure and preserve it as the source of truth.
2. Inspect the v2 handoff archive and extract its inventory and raw files where possible.
3. Create a source-file inventory with filename, dataset, format, size and validation status.
4. Compare actual files with the intended dataset inventory.
5. Re-run source inspection using v4.
6. Fix parser defects and source mappings only where evidence supports the change.
7. Re-run data preparation with v4.
8. Validate SOH calculations, capacity units, battery IDs, cycle counts and missing metadata.
9. Generate the final unified dataset and quality report.
10. Re-run the synthetic test suite.
11. Confirm that no v2-only processed output is being used as trusted v4 data.
12. Keep the Streamlit dashboard read-only and connected to actual prepared files where available.
13. Wait for the user's explicit approval before proceeding to new model implementation or training.

# 16. Definition of a properly prepared project handoff

The updated project should contain:

- The v4 source code.
- Existing tests and project documentation.
- `docs/LLM_PROJECT_CONTEXT.md` — this file.
- `docs/PRESENTATION_HANDOFF.md` — the separate presentation brief.
- `streamlit_app.py` — a read-only research progress/data explorer.
- `requirements-streamlit.txt`.
- Clear dashboard run instructions.
- No fabricated results.
- No accidental replacement of v4 code with v2 code.
- A clear distinction between verified data, previously inspected data and pending validation.

**The immediate objective is to make the existing work understandable, reproducible and presentable. The next research phase begins only after the data foundation is verified and the user approves moving forward.**