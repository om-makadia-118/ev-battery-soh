# Dataset acquisition and validation plan

This document is specific to the dataset files the team currently has. Do not assume the inventory printed in the initial report is accurate until each downloaded source is counted and validated.

## Current inventory and the required action

| Dataset | Files currently reported by the team | Verified interpretation | Next action |
|---|---|---|---|
| NASA PCoE | `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` | 4 cells, 2 Ah nominal, LCO, discharge capacity in Ah; original source EOL is 70% | Keep all 4. The report's table saying 28 NASA batteries does not match these four files. Correct the report or acquire a separately identified NASA collection. |
| Oxford Dataset 1 | `Oxford_Battery_Degradation_Dataset_1.mat` | 8 Kokam LCO pouch cells, 0.74 Ah, 40°C, C1 discharge characterization approximately every 100 drive cycles | Keep it. Python can read `.mat`; the loader supports MATLAB v5 and v7.3/HDF5. |
| XJTU | 15 `.pkl` files | If names are `XJTU_3C_battery-1.pkl` through `-15.pkl`, these are the 15 cells in Batch 2 (3C), not the full 55-cell dataset | Keep all 15 for a pilot. For the report's full 55 cells, download the official 2.4 GB `Battery Dataset.zip` from Zenodo. |
| CALCE | All 15 CS2 archives downloaded; one CS2_33 Arbin workbook and one CS2_8 CADEX text file inspected | 15 CS2 LCO cells; multiple date-named files per cell. Native parsers support Arbin Excel and CADEX tab-separated logs. The CADEX reader was validated against 11 discharge segments in the supplied CS2_8 file; CS2_21 still needs a real-file spot check. | Extract all archives under `data/raw/calce/`, retaining filenames. Run `inspect` on one CS2_21 text file and then prepare. |
| MATR | Not downloaded | Severson et al. 2019 dataset: 124 LFP/graphite cells, nominal 1.1 Ah, distributed in three original batch `.mat` files | Download the three named batch files from the official TRI data project. |

## 1. NASA

Place the four files here:

```text
data/raw/nasa/
    B0005.mat
    B0006.mat
    B0007.mat
    B0018.mat
```

The NASA readme provided by the team states that capacity is stored on discharge operations in `cycle.data.Capacity` (Ah), and the four cells reach the source's 70% EOL criterion (2 Ah to 1.4 Ah). We derive SOH as `100 * capacity_ah / 2.0`. The project-wide 80% threshold is an analysis threshold; do not overwrite the dataset's source EOL metadata or remove observations below 80% by default.

**Count correction:** these four files represent four cells, not 28. The report's NASA count of 28 cannot be supported by this selected file set.

## 2. Oxford

Place the file here:

```text
data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat
```

This dataset contains eight 740 mAh Kokam pouch cells, tested at 40°C. The main file stores characterization measurements roughly every 100 drive cycles. The project reads `CellN/cycNNNN/C1dc/q`; `q` is in mAh, and the range of the measured discharge charge vector is converted to Ah. We keep the actual checkpoint cycle number and do not fabricate the intervening 99 cycles.

A `.mat` extension does not mean MATLAB is required. The project uses `scipy.io.loadmat` for traditional MAT files and `h5py` for HDF5/MATLAB v7.3 files. To inspect:

```powershell
python -m battery_soh inspect --path "data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat"
```

Then `python -m battery_soh prepare --raw data/raw` will attempt the native reader.

## 3. XJTU

### The 15 pickle files

If your files are named `XJTU_3C_battery-1.pkl` to `XJTU_3C_battery-15.pkl`, use **all 15**. They represent Batch 2 (3C), not a random choice among files. This is a valid 15-cell subset for a pilot, but it is not the full 55-cell XJTU dataset and is not the complete XJTU row in the report.

Place them in:

```text
data/raw/xjtu/
    XJTU_3C_battery-1.pkl
    ...
    XJTU_3C_battery-15.pkl
```

The project now has a native reader for the standardized BatteryLife pickle format (`cell_id`, `cycle_data`, `nominal_capacity_in_Ah`, and per-cycle `discharge_capacity_in_Ah` traces). It extracts one capacity label per observed cycle using the range of the cumulative discharge-capacity trace. The loader does not assume the object is a DataFrame.

Inspect one file first, then run `prepare`:

```powershell
python -m battery_soh inspect --path "data/raw/xjtu/XJTU_3C_battery-1.pkl"
python -m battery_soh prepare --raw data/raw
```

Only unpickle files obtained from the official dataset / known research release. Pickle files can execute code when loaded. If inspection shows a different schema or preparation reports missing keys, send the output so the parser can be adjusted against the exact release.

### Full 55-cell dataset

The official XJTU description lists six batches: 8 + 15 + 8 + 8 + 8 + 8 = 55 cells. The official Zenodo release provides `Battery Dataset.zip` (2.4 GB). The current BatteryLife repository also announces a complete standardized XJTU release in its v12 data updates. Download the raw archive if you need the original 55-cell batch files:

- https://zenodo.org/records/10963339
- Dataset description and batch conditions: https://wang-fujin.github.io/
- Original preprocessing code: https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library

Extract the six batches and retain original batch and cell IDs. Do not merge cells from different batches just because their local cell number repeats.

## 4. CALCE — download CS2, not Storage Data

The report's stated selection (15 cells, 1.1 Ah, LCO) corresponds to the CALCE **CS2 prismatic cell cycling collection**. It is not the 144-cell Storage Data experiment.

Official source: https://calce.umd.edu/battery-data (section **CS2 Battery**).

The 15 CS2 cells listed by CALCE are:

```text
CS2_3, CS2_5, CS2_6, CS2_7, CS2_8,
CS2_9, CS2_21, CS2_24, CS2_25,
CS2_33, CS2_34, CS2_35, CS2_36, CS2_37, CS2_38
```

All 15 archives have now been downloaded by the team. Extract each archive under `data/raw/calce/`, retaining the original filenames. A convenient layout is:

```text
data/raw/calce/
    CS2_3/
        CS2_3_*.xlsx
    CS2_5/
        CS2_5_*.xlsx
    ...
    CS2_8/
        CS2_8_*.txt
    CS2_21/
        CS2_21_*.txt
    ...
    CS2_38/
        CS2_38_*.xlsx
```

CALCE states that each cell is represented by multiple date-named test files. Most are Arbin Excel workbooks with `Channel_*` sheets; CS2_8 and CS2_21 were recorded using a CADEX tester and are supplied as tab-separated `.txt` files. The native parser supports both formats. The CADEX reader has been checked against the supplied `CS2_8_1_19_10.txt` file; inspect one CS2_21 file to confirm its header and status codes match before treating both CADEX cells as fully validated.

The Arbin parser combines all date-named files per cell, sorts the raw records chronologically, and rebases `Cycle_Index` when it restarts in a later file. It extracts per-cycle discharge capacity from the cumulative `Discharge_Capacity(Ah)` counter. It only keeps full-capacity checks whose cycle includes a full-charge voltage (at least 4.15 V) and a discharge endpoint (at most 2.85 V). This keeps deliberate partial cycling in Type 5/6 out of the SOH labels; Type 4 cycles with a raised cutoff are also excluded unless they reach the full-discharge cutoff. The CADEX parser identifies discharge steps using `Status category == 6`, integrates absolute current (mA) over elapsed time (seconds) to obtain Ah, and uses the observed 0–100 `Capacity` field only as a discharge-completion heuristic (final value at least 95), never as an Ah measurement. It merges date-named text files and rebases reset discharge counters. Original cycle chronology and file paths are preserved.

**Rated-capacity caveat:** the CALCE website lists 1.1 Ah as the general CS2 capacity rating, which is the value used in the project report and parser default. The supplied CS2_8 CADEX file yields integrated discharge capacities of about 1.10–1.12 Ah across its first 11 observed cycles, consistent with 1.1 Ah for this cell. The first available CS2_33 workbook instead records about 0.891 Ah; that may reflect degradation before this recording began, a cell-specific rating, or another protocol/source detail. The file alone cannot distinguish these explanations. Keep the report's 1.1 Ah assumption for now, and investigate the source metadata before final SOH claims; do not infer a different rated capacity solely from the first observed cycle.

After extraction, run:

```powershell
python -m battery_soh inspect --path "data/raw/calce/CS2_33/CS2_33_1_10_11.xlsx"
python -m battery_soh inspect --path "data/raw/calce/CS2_8/CS2_8_1_19_10.txt"
python -m battery_soh prepare --raw data/raw
```

The 144-cell Storage dataset is a separate storage-aging experiment with tests at different storage SOCs and temperatures. It can support a different research question, but it should not silently replace the report's CS2 cycling dataset.

## 5. MATR / MIT–Stanford–Toyota

The report refers to the Severson et al. (2019) fast-charging cycle-life dataset: 124 LFP/graphite A123 cells, nominal capacity about 1.1 Ah.

Official TRI project:

https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

On the project page, open **Data-driven prediction of battery cycle life before capacity degradation** and download these three batch files:

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

They are large MATLAB v7.3/HDF5 files. Use `h5py`; MATLAB is not required. The current MATR parser reads summary discharge-capacity records, but batch/cell identity and any cross-batch continuity must be checked before final train/test splits. Never assume a per-file cell index is a globally unique physical battery without checking source metadata.

## 6. Inspect the files you already have

From the project root in PowerShell:

```powershell
python -m battery_soh inspect --path "data/raw/nasa/B0005.mat"
python -m battery_soh inspect --path "data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat"
python -m battery_soh inspect --path "data/raw/xjtu/XJTU_3C_battery-1.pkl"
```

Or run:

```powershell
.\scripts\inspect_downloads.ps1
```

The supplied XJTU pickle inspection output matches the native reader, and the CS2_33 workbook plus CS2_8 CADEX log have both been inspected. Before running the full CALCE preparation, inspect one CS2_21 `.txt` file to confirm it uses the same CADEX columns and status codes. MATR batch identity still requires source-level validation.

## 7. Canonical output and validation

The canonical table is one row per **observed capacity-check event**:

```text
dataset,battery_id,cycle,capacity_ah,rated_capacity_ah,temperature_c,chemistry,source_file
```

The SOH label is:

```text
SOH (%) = 100 * measured_discharge_capacity_Ah / rated_capacity_Ah
```

The current-cycle capacity and SOH are targets, never predictor inputs. The first working predictor uses cycle number only as an explicitly limited age baseline. Later physical features may be added only after confirming that they are measured and comparable across all five domains.

Do not interpolate target labels for unmeasured cycles. Do not mix partial discharge capacity with full-discharge capacity as if they measured the same thing. Do not use battery cycles from one cell in both train and test. Use 80% as the common analysis EOL threshold only after preserving the original source EOL information.
