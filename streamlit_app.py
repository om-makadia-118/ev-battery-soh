
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="EV Battery SOH Research",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #10243a, #173c50);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.35rem;
    }
    .hero p {
        color: #d5e3ed;
        margin-bottom: 0;
    }
    .status-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        min-height: 125px;
    }
    .muted {
        color: #8a94a3;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Dataset inventory
# These are previously identified
# inventory figures, not a claim
# of complete v4 validation.
# -----------------------------
DATASETS = [
    {
        "Dataset": "NASA",
        "Chemistry": "LCO",
        "Nominal capacity": "2 Ah",
        "Reported scope": "28",
        "Identified local scope": "4",
        "Files / format": "MATLAB (.mat)",
        "Progress": "Four cells identified; final v4 QC pending",
        "Notes": "B0005, B0006, B0007, B0018",
    },
    {
        "Dataset": "CALCE",
        "Chemistry": "LCO",
        "Nominal capacity": "1.1 Ah",
        "Reported scope": "15",
        "Identified local scope": "15 CS2",
        "Files / format": "Arbin Excel and CADEX text",
        "Progress": "Representative files investigated; full validation pending",
        "Notes": "CS2 cycling data",
    },
    {
        "Dataset": "XJTU",
        "Chemistry": "NCM",
        "Nominal capacity": "2 Ah",
        "Reported scope": "55",
        "Identified local scope": "15",
        "Files / format": "Pickle (.pkl)",
        "Progress": "Batch 2, 3C sample structure inspected",
        "Notes": "Not the full 55-cell collection",
    },
    {
        "Dataset": "Oxford",
        "Chemistry": "LCO",
        "Nominal capacity": "0.74 Ah",
        "Reported scope": "8",
        "Identified local scope": "8",
        "Files / format": "MATLAB (.mat)",
        "Progress": "Sparse characterization records; no interpolation",
        "Notes": "40°C testing condition",
    },
    {
        "Dataset": "MATR",
        "Chemistry": "LFP",
        "Nominal capacity": "1.1 Ah",
        "Reported scope": "124",
        "Identified local scope": "124 targeted",
        "Files / format": "Three batch MAT files",
        "Progress": "Cell identity and batch mapping need verification",
        "Notes": "MIT-Stanford-Toyota collection",
    },
]

# -----------------------------
# Data discovery
# -----------------------------
UNIFIED_CANDIDATES = [
    ROOT / "data" / "processed" / "unified.csv",
    ROOT / "data" / "processed" / "canonical.csv",
    ROOT / "data" / "processed" / "unified_dataset.csv",
]

QUALITY_CANDIDATES = [
    ROOT / "data" / "processed" / "data_quality.csv",
    ROOT / "data" / "processed" / "quality_report.csv",
]


def first_existing(paths):
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv(path_string):
    return pd.read_csv(path_string)


def find_column(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_unified_data(df):
    data = df.copy()

    dataset_col = find_column(
        data, ["dataset", "source", "dataset_name"]
    )
    battery_col = find_column(
        data, ["battery_id", "cell_id", "battery", "cell"]
    )
    cycle_col = find_column(
        data, ["cycle_index", "cycle_number", "cycle"]
    )
    soh_col = find_column(
        data, ["soh_percent", "soh", "state_of_health"]
    )
    capacity_col = find_column(
        data,
        ["capacity_ah", "discharge_capacity_ah", "capacity"],
    )

    return data, {
        "dataset": dataset_col,
        "battery": battery_col,
        "cycle": cycle_col,
        "soh": soh_col,
        "capacity": capacity_col,
    }


def show_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metric_row():
    a, b, c, d = st.columns(4)
    a.metric("Datasets in scope", "5")
    b.metric("Chemistry families", "3")
    c.metric("Real benchmark results", "Not yet available")
    d.metric("Current phase", "Data foundation")


def show_pipeline():
    steps = [
        ("01", "Raw datasets", "MATLAB, Pickle, Excel, CADEX"),
        ("02", "Dataset readers", "Source-specific parsing"),
        ("03", "Normalization", "Units, IDs and cycle mapping"),
        ("04", "SOH calculation", "Discharge capacity / rated capacity"),
        ("05", "Quality checks", "Missing values, units and integrity"),
        ("06", "Unified dataset", "Common representation for evaluation"),
    ]

    cols = st.columns(3)

    for index, item in enumerate(steps):
        number, title, description = item
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="muted">{number}</div>
                    <h4>{title}</h4>
                    <div>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if index in (2,):
            st.write("")


def show_inventory():
    st.subheader("Dataset registry")
    st.caption(
        "Inventory figures reflect previously identified project scope. "
        "They do not imply complete validation through v4."
    )

    df = pd.DataFrame(DATASETS)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🔋 EV Battery SOH")
st.sidebar.caption("Cross-dataset generalization research")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Registry",
        "Data Pipeline",
        "Data Explorer",
        "Validation",
        "Research Plan",
    ],
)

unified_path = first_existing(UNIFIED_CANDIDATES)
quality_path = first_existing(QUALITY_CANDIDATES)

real_df = None
column_map = {}

if unified_path:
    try:
        raw_df = read_csv(str(unified_path))
        real_df, column_map = normalize_unified_data(raw_df)
    except Exception as exc:
        st.sidebar.error(f"Could not load unified CSV: {exc}")

st.sidebar.divider()
st.sidebar.caption(
    "Read-only research dashboard. No model training is performed here."
)


# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    show_header(
        "EV Battery State of Health",
        "Cross-dataset generalization of machine-learning models",
    )

    show_metric_row()

    st.subheader("Research objective")
    st.write(
        "Investigate whether battery SOH prediction models trained on "
        "one dataset can generalize to batteries from other datasets, "
        "chemistries and testing conditions."
    )

    st.subheader("Project progress")
    show_pipeline()

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("### Engineering foundation")
        st.markdown(
            """
            - Five public datasets selected.
            - Multiple source formats investigated.
            - Dataset-specific parsing work established.
            - Common SOH calculation defined.
            - Data-quality and leakage safeguards designed.
            - Existing experiment framework documented.
            """
        )

    with right:
        st.markdown("### Current focus")
        st.markdown(
            """
            - Revalidate raw files using v4.
            - Finalize source mappings.
            - Generate the verified unified dataset.
            - Review data-quality reports.
            - Prepare for empirical benchmarking.
            """
        )

    st.info(
        "No real-data model accuracy results are presented in this "
        "dashboard. The current focus is data and pipeline validation."
    )

# -----------------------------
# Dataset registry
# -----------------------------
elif page == "Dataset Registry":
    show_header(
        "Dataset Registry",
        "Five public sources with different chemistries and data formats",
    )

    show_inventory()

    st.subheader("Important inventory corrections")
    st.warning(
        "The initial review listed 28 NASA and 55 XJTU batteries. "
        "The currently identified local scope is four NASA files and "
        "15 XJTU Batch 2 files. These figures must not be presented as "
        "full-dataset coverage."
    )

    st.subheader("Why dataset-specific processing is needed")
    st.write(
        "The sources contain different record structures, units, "
        "measurement intervals and capacity definitions. A common "
        "representation requires source-aware parsing and validation."
    )

# -----------------------------
# Data pipeline
# -----------------------------
elif page == "Data Pipeline":
    show_header(
        "Data Processing Architecture",
        "From heterogeneous battery files to a common SOH representation",
    )

    show_pipeline()

    st.subheader("SOH calculation")

    st.latex(
        r"SOH(\%) = "
        r"\frac{\text{Measured discharge capacity (Ah)}}"
        r"{\text{Rated capacity (Ah)}} \times 100"
    )

    st.markdown(
        """
        **Processing principles**

        - Convert source units explicitly.
        - Preserve battery and dataset identifiers.
        - Use measured discharge capacity.
        - Preserve values above 100% when supported by measurements.
        - Retain sparse characterization observations.
        - Do not interpolate unobserved SOH labels.
        - Keep current-cycle capacity and SOH out of predictor features.
        """
    )

    st.subheader("Example: CALCE CADEX")
    st.markdown(
        """
        The inspected CADEX structure required special handling:

        - Status 3: charging.
        - Status 6: discharging.
        - Voltage: mV to V.
        - Current: mA to A.
        - Duration: seconds.
        - Capacity field: progress percentage, not Ah.
        - Discharge Ah: calculated by integrating discharge current over time.
        """
    )

# -----------------------------
# Data explorer
# -----------------------------
elif page == "Data Explorer":
    show_header(
        "Unified Data Explorer",
        "Visualize the prepared battery observations when available",
    )

    if real_df is not None:
        st.success(f"Loaded: {unified_path.relative_to(ROOT)}")

        st.metric("Rows", f"{len(real_df):,}")
        st.metric("Columns", len(real_df.columns))

        st.subheader("Dataset preview")
        st.dataframe(real_df.head(30), use_container_width=True)

        dataset_col = column_map["dataset"]
        battery_col = column_map["battery"]
        cycle_col = column_map["cycle"]
        soh_col = column_map["soh"]

        if soh_col and cycle_col:
            plot_df = real_df.copy()

            if dataset_col:
                datasets = sorted(
                    plot_df[dataset_col].dropna().astype(str).unique()
                )
                selected_dataset = st.selectbox(
                    "Dataset",
                    ["All"] + datasets,
                )

                if selected_dataset != "All":
                    plot_df = plot_df[
                        plot_df[dataset_col].astype(str) == selected_dataset
                    ]

            if battery_col:
                batteries = sorted(
                    plot_df[battery_col].dropna().astype(str).unique()
                )
                if batteries:
                    selected_battery = st.selectbox(
                        "Battery",
                        ["All"] + batteries,
                    )
                    if selected_battery != "All":
                        plot_df = plot_df[
                            plot_df[battery_col].astype(str)
                            == selected_battery
                        ]

            plot_df[cycle_col] = pd.to_numeric(
                plot_df[cycle_col], errors="coerce"
            )
            plot_df[soh_col] = pd.to_numeric(
                plot_df[soh_col], errors="coerce"
            )
            plot_df = plot_df.dropna(subset=[cycle_col, soh_col])

            if plot_df.empty:
                st.warning("No valid cycle/SOH observations to plot.")
            else:
                if plot_df[soh_col].median() <= 1.5:
                    st.caption(
                        "SOH appears to be stored as a fraction; "
                        "displaying it as a percentage."
                    )
                    plot_df[soh_col] = plot_df[soh_col] * 100

                color = battery_col if battery_col else dataset_col

                fig = px.line(
                    plot_df.sort_values(cycle_col),
                    x=cycle_col,
                    y=soh_col,
                    color=color if color else None,
                    title="Observed SOH across characterization records",
                    labels={
                        cycle_col: "Cycle / characterization index",
                        soh_col: "SOH (%)",
                    },
                )
                fig.update_layout(
                    legend_title_text="Battery / dataset",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "The CSV is loaded, but recognizable cycle and SOH "
                "columns were not found. Review the schema mapping."
            )

    else:
        st.warning(
            "No recognized unified real-data CSV was found. "
            "Expected location: data/processed/unified.csv"
        )

        st.write(
            "Once the v4 preparation pipeline generates the unified CSV, "
            "this page will automatically display its rows and SOH trends."
        )

        show_example = st.checkbox(
            "Show an illustrative SOH trajectory "
            "(synthetic example, not experimental data)",
            value=False,
        )

        if show_example:
            rng = np.random.default_rng(42)
            cycles = np.arange(1, 101)
            soh = 100 - 0.16 * cycles + rng.normal(0, 0.45, len(cycles))

            example = pd.DataFrame(
                {
                    "Cycle": cycles,
                    "SOH (%)": soh,
                }
            )

            st.info(
                "Illustrative synthetic data only. This is not a result "
                "from NASA, CALCE, XJTU, Oxford or MATR."
            )

            fig = px.line(
                example,
                x="Cycle",
                y="SOH (%)",
                title="Illustrative battery degradation trajectory",
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Validation
# -----------------------------
elif page == "Validation":
    show_header(
        "Data Integrity and Validation",
        "Safeguards required for scientifically reliable comparisons",
    )

    checks = [
        ("Capacity units", "Verify Ah, mAh, A, mA, V and mV conversions."),
        ("Capacity semantics", "Confirm that each capacity field represents measured Ah."),
        ("Battery identifiers", "Ensure cell identities remain unique and traceable."),
        ("Cycle mapping", "Preserve original cycle and characterization references."),
        ("Missing records", "Report missing values and parser failures."),
        ("SOH calculation", "Validate rated capacity and discharge capacity."),
        ("EOL definitions", "Preserve source thresholds and identify common-threshold analyses."),
        ("Leakage prevention", "Split whole batteries, not individual cycle rows."),
        ("Feature leakage", "Exclude current-cycle capacity and SOH from predictors."),
        ("Oxford sampling", "Do not interpolate unobserved characterization labels."),
    ]

    for title, description in checks:
        st.markdown(f"- **{title}:** {description}")

    st.subheader("Generated quality report")

    if quality_path:
        try:
            quality_df = read_csv(str(quality_path))
            st.success(f"Found: {quality_path.relative_to(ROOT)}")
            st.dataframe(quality_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read quality report: {exc}")
    else:
        st.info(
            "A current v4 data-quality report has not been found at "
            "data/processed/data_quality.csv."
        )

    st.caption(
        "The checks above describe the validation protocol. They should "
        "not be interpreted as evidence that every check has passed."
    )

# -----------------------------
# Research plan
# -----------------------------
elif page == "Research Plan":
    show_header(
        "Research and Experiment Plan",
        "The benchmark framework planned after data validation",
    )

    st.subheader("Model families represented in the existing framework")

    left, right = st.columns(2)

    with left:
        st.markdown("### Classical")
        st.markdown(
            """
            - Mean baseline
            - Ridge Regression
            - ElasticNet
            - XGBoost
            """
        )

    with right:
        st.markdown("### Deep learning")
        st.markdown(
            """
            - LSTM
            - 1D-CNN
            - Transformer
            """
        )

    st.caption(
        "These model families are documented in the existing project "
        "framework. This dashboard does not train them or display "
        "real-data benchmark scores."
    )

    st.subheader("Evaluation protocols")

    protocols = [
        (
            "Within-dataset",
            "Train and test on different batteries from the same dataset.",
        ),
        (
            "Cross-dataset",
            "Train on one dataset and test on a different dataset.",
        ),
        (
            "Leave-one-dataset-out",
            "Train on four datasets and test on the held-out fifth dataset.",
        ),
        (
            "Few-shot adaptation",
            "Use a small number of target support batteries.",
        ),
        (
            "CORAL",
            "Align permitted source and target-support feature distributions.",
        ),
    ]

    for name, description in protocols:
        st.markdown(f"- **{name}:** {description}")

    st.subheader("Planned outputs")
    st.markdown(
        """
        - 5×5 cross-dataset transfer matrix.
        - MAE, RMSE and MAPE comparisons.
        - Model ranking.
        - Error analysis by chemistry, temperature and charging speed,
          where source metadata supports it.
        - Few-shot data-needed curves.
        - Failure analysis and reproducible experiment summaries.
        """
    )

    st.warning(
        "Model implementation/training is intentionally not part of "
        "the current project-cleanup phase."
    )

# -----------------------------
# Footer
# -----------------------------
st.sidebar.divider()
st.sidebar.caption("EV Battery SOH Research | Presentation Dashboard")
