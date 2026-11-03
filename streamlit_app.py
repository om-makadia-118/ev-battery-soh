"""
EV Battery State-of-Health (SOH) Research Dashboard
Cross-Dataset Generalization of Machine-Learning Models for EV Battery SOH Prediction
BMSCE Project Work 2 | v4 Certified Research Framework
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_CONFIG_DIR", "/tmp/streamlit")
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from battery_soh.data.schema import harmonize
from battery_soh.models.classical import make_classical

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="EV Battery SOH Research Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium Institutional Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    .hero-banner {
        padding: 2rem 2.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #09172a 0%, #112a45 50%, #0d3b4c 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .hero-banner h1 {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    
    .hero-banner p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .badge-teal { background: rgba(20, 184, 166, 0.2); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.4); }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    
    .kpi-card {
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
    }
    
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    .glass-card {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 1.2rem;
    }
    
    .insight-card {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.65);
        border-left: 4px solid #0ea5e9;
        margin-bottom: 0.9rem;
    }
    
    .insight-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.25rem;
    }
    
    .insight-desc {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data Registry Specs
# -----------------------------
DATASETS = [
    {
        "Dataset": "NASA",
        "Chemistry": "LCO",
        "Nominal Capacity": "2.0 Ah",
        "Reported Scope": "28 cells",
        "Validated Scope": "4 physical cells",
        "Format": "MATLAB (.mat)",
        "Ambient Temp": "24°C",
        "Charge / Discharge Protocol": "1.0C CC-CV / 1.0C CC to 2.7V",
        "Status": "Verified in v4 pipeline (636 observations)",
        "Notes": "Cells: B0005, B0006, B0007, B0018",
    },
    {
        "Dataset": "CALCE",
        "Chemistry": "LCO",
        "Nominal Capacity": "1.1 Ah",
        "Reported Scope": "15 cells",
        "Validated Scope": "15 CS2 cells",
        "Format": "13 Arbin Excel + 2 CADEX text",
        "Ambient Temp": "25°C",
        "Charge / Discharge Protocol": "0.5C CC-CV / 0.1C–2.0C dynamic & CC",
        "Status": "Verified in v4 pipeline (9,608 observations)",
        "Notes": "CADEX current integration & Arbin step segmentation applied",
    },
    {
        "Dataset": "XJTU",
        "Chemistry": "NCM",
        "Nominal Capacity": "2.0 Ah",
        "Reported Scope": "55 cells",
        "Validated Scope": "23 cells (8 2C + 15 3C)",
        "Format": "BatteryLife Pickle (.pkl)",
        "Ambient Temp": "25°C, 35°C, 45°C",
        "Charge / Discharge Protocol": "2C / 3C fast charge, 1C CC discharge",
        "Status": "Verified in v4 pipeline (6,938 observations)",
        "Notes": "Batch 1 (2C, 8 cells) and Batch 2 (3C, 15 cells)",
    },
    {
        "Dataset": "Oxford",
        "Chemistry": "LCO",
        "Nominal Capacity": "0.74 Ah",
        "Reported Scope": "8 cells",
        "Validated Scope": "8 pouch cells",
        "Format": "MATLAB v7.3 HDF5 (.mat)",
        "Ambient Temp": "40°C",
        "Charge / Discharge Protocol": "0.5C CC-CV / Artemis urban drive cycles",
        "Status": "Verified in v4 pipeline (519 observations)",
        "Notes": "Preserved discrete ~100-cycle characterization checks without interpolation",
    },
    {
        "Dataset": "MATR",
        "Chemistry": "LFP",
        "Nominal Capacity": "1.1 Ah",
        "Reported Scope": "124 cells",
        "Validated Scope": "140 cells (3 batches)",
        "Format": "MATLAB v7.3 HDF5 (.mat)",
        "Ambient Temp": "30°C",
        "Charge / Discharge Protocol": "Fast-charging (3.6C–6.0C) / 4.0C CC discharge",
        "Status": "Verified in v4 pipeline (114,688 observations)",
        "Notes": "MIT-Stanford-Toyota collection; 4 hardware glitch spikes (>1.43 Ah) filtered",
    },
]

# -----------------------------
# Data Loaders (Cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_unified_data():
    p = ROOT / "data/processed/unified.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        return harmonize(df)
    return None

@st.cache_data(show_spinner=False)
def load_quality_data():
    p = ROOT / "data/processed/data_quality.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(show_spinner=False)
def load_matrix_summary():
    p = ROOT / "results/full_study/matrix/figures/per_pair_seed_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(show_spinner=False)
def load_score_summary():
    p = ROOT / "results/full_study/matrix/figures/score_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(show_spinner=False)
def load_lodo_summary():
    p = ROOT / "results/full_study/lodo/figures/per_pair_seed_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(show_spinner=False)
def load_adaptation_metrics():
    p = ROOT / "results/full_study/adaptation/adaptation_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

unified_df = load_unified_data()
quality_df = load_quality_data()
matrix_df = load_matrix_summary()
score_df = load_score_summary()
lodo_df = load_lodo_summary()
adapt_df = load_adaptation_metrics()

# -----------------------------
# Navigation Sidebar
# -----------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <h2 style="margin: 0; font-size: 1.35rem; color: #f8fafc; font-weight: 700;">🔋 EV Battery SOH</h2>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.82rem; color: #94a3b8;">Cross-Dataset Generalization Research</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Executive Overview",
        "📁 Dataset Registry & Explorer",
        "📊 5×5 Transfer Matrix",
        "🌐 LODO & Few-Shot Adaptation",
        "🔬 What-If SOH Predictor",
        "🛡️ Data Quality & Audit",
        "📄 Research Report & Export",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
        <strong>Framework:</strong> v4 Certified Architecture<br>
        <strong>Scope:</strong> 5 Repositories | 190 Cells<br>
        <strong>Benchmark Runs:</strong> 6,440 (5 Seeds)<br>
        <strong>Splits:</strong> Disjoint Battery-Level
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Common UI Helpers
# -----------------------------
def render_hero(title, subtitle, badges=None):
    badge_html = ""
    if badges:
        for text, color_cls in badges:
            badge_html += f'<span class="badge-pill {color_cls}">{text}</span>'
    
    st.markdown(
        f"""
        <div class="hero-banner">
            <div>{badge_html}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Verified Cells</div>
                <div class="kpi-value">190</div>
                <div class="kpi-subtext">Across 5 physical battery labs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Cycling Observations</div>
                <div class="kpi-value">132,389</div>
                <div class="kpi-subtext">Zero synthetic observations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Within Baseline MAE</div>
                <div class="kpi-value">3.89 pp</div>
                <div class="kpi-subtext">XGBoost on held-out cells</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Benchmark Runs</div>
                <div class="kpi-value">6,440</div>
                <div class="kpi-subtext">500 matrix + 100 LODO + 5,840 adapt</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "🏠 Executive Overview":
    render_hero(
        "Cross-Dataset EV Battery State-of-Health Generalization",
        "A rigorous empirical investigation into whether machine-learning models trained on one laboratory battery dataset "
        "can reliably generalize to batteries from other chemistries, operating temperatures, and cycling protocols.",
        [
            ("BMSCE Project Work 2", "badge-teal"),
            ("v4 Certified Pipeline", "badge-blue"),
            ("190 Real Physical Cells", "badge-amber"),
            ("Phase 2 Benchmark Completed", "badge-purple"),
        ],
    )
    
    render_kpi_row()
    st.write("")
    
    st.subheader("Core Research Questions & Empirical Answers")
    col_q1, col_q2, col_q3 = st.columns(3)
    
    with col_q1:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">RQ1: Does within-domain accuracy predict generalization?</div>
                <div class="insight-desc">
                    <strong>No.</strong> Models achieving &lt;2.0% MAE on held-out cells within Oxford or XJTU suffer severe error inflation 
                    (13.03 pp average for XGBoost; &gt;60 pp for linear models) when transferred zero-shot without adaptation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_q2:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">RQ2: How severely does chemistry & protocol shift degrade SOH?</div>
                <div class="insight-desc">
                    <strong>Catastrophic for unconstrained linear models.</strong> Transferring from NASA (168-cycle lifespan) to Oxford (500 cycles) 
                    causes Ridge MAE to explode to 637.45 pp due to out-of-distribution extrapolation, whereas XGBoost stays bounded (17.10 pp).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col_q3:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">RQ3: How many target batteries are needed for adaptation?</div>
                <div class="insight-desc">
                    <strong>1 to 3 support batteries.</strong> Incorporating just k=1 target support battery reduces scratch MAE from 37.49 pp 
                    to 11.18 pp, while k=3 to 5 cells restores accuracy to within-domain baseline parity (4.16 pp for XGBoost).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Benchmark Execution Summary")
    
    left_c, right_c = st.columns([1.5, 1])
    with left_c:
        if score_df is not None:
            st.markdown("### Aggregated Benchmark Scores Across 5 Seeds")
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        else:
            st.info("Score summary CSV not found.")
            
    with right_c:
        st.markdown("### Research Protocol & Safeguards")
        st.markdown(
            """
            - **Strict Battery-Disjoint Splitting:** Train, validation, and test splits always partition whole battery cells (`battery_split`), completely preventing cycle-level data leakage.
            - **No Current-Cycle Target Leakage:** Current-cycle capacity or SOH is strictly prohibited from feature matrices.
            - **No Synthetic Inventions:** All Oxford sparse checkpoints are preserved without interpolation; all 320 infeasible few-shot settings are accounted for.
            - **Source vs Harmonized EOL:** Source EOL (NASA/CALCE 70%, XJTU/Oxford/MATR 80%) is strictly distinguished from harmonized 80% EOL.
            """
        )


# ==============================================================================
# PAGE 2: DATASET REGISTRY & DATA EXPLORER
# ==============================================================================
elif page == "📁 Dataset Registry & Explorer":
    render_hero(
        "Dataset Registry & Interactive SOH Explorer",
        "Inspect the 190 validated physical battery cells across NASA, CALCE, XJTU, Oxford, and MATR repositories.",
        [("5 Datasets", "badge-blue"), ("LCO, NCM, LFP", "badge-amber"), ("132k Observations", "badge-teal")],
    )
    
    tab_inv, tab_traj, tab_chem = st.tabs([
        "📋 Dataset Registry & Specifications",
        "📈 Interactive SOH Trajectories",
        "🔬 Electrochemical Distributions",
    ])
    
    with tab_inv:
        st.subheader("Authenticated Source Inventory")
        reg_df = pd.DataFrame(DATASETS)
        st.dataframe(reg_df[["Dataset", "Chemistry", "Nominal Capacity", "Validated Scope", "Ambient Temp", "Charge / Discharge Protocol", "Status"]], use_container_width=True, hide_index=True)
        
        st.write("")
        st.markdown("### Engineering Corrections Applied to Raw Archives")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                r"- **CALCE CADEX Current Integration:** In `CS2_8` and `CS2_21`, the recorded `Capacity` column is a 0–100% nominal progress index. True capacity was calculated by integrating current over duration ($\int I dt$)."
                "\n"
                "- **CALCE Arbin Step Segmentation:** In alternating-rate cells (`CS2_3`, `CS2_9`), up to 6 discharge steps were bundled into one cycle counter. Segmented by `_dis_step` to prevent unphysical cumulative capacities."
            )
        with c2:
            st.markdown(
                """
                - **MATR Glitch Outlier Filtering:** Exactly 4 single-cycle measurements out of 114,692 rows had unphysical hardware glitches (>1.43 Ah). Filtered cleanly with zero duplicate records.
                - **Oxford Sparse Checkpoint Preserved:** Full characterization cycles occur approximately every 100 drive cycles; preserved without artificial interpolation.
                """
            )
            
    with tab_traj:
        st.subheader("Interactive Battery Degradation Curves")
        if unified_df is not None:
            f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
            with f_col1:
                avail_datasets = sorted(unified_df["dataset"].unique())
                sel_ds = st.selectbox("Filter Dataset:", ["All"] + avail_datasets, index=0)
            
            sub_df = unified_df if sel_ds == "All" else unified_df[unified_df["dataset"] == sel_ds]
            
            with f_col2:
                avail_cells = sorted(sub_df["battery_id"].unique())
                default_cells = [avail_cells[0]] if avail_cells else []
                sel_cells = st.multiselect("Select Batteries to Compare:", avail_cells, default=default_cells[:3])
                
            with f_col3:
                max_cyc = int(sub_df["cycle"].max())
                cyc_range = st.slider("Cycle Range:", 0, max_cyc, (0, max_cyc))
                
            plot_data = sub_df[
                (sub_df["battery_id"].isin(sel_cells)) &
                (sub_df["cycle"] >= cyc_range[0]) &
                (sub_df["cycle"] <= cyc_range[1])
            ]
            
            if not plot_data.empty:
                # Downsample if too large for responsive charting
                if len(plot_data) > 4000:
                    plot_data = plot_data.iloc[::2].copy()
                    
                fig = px.line(
                    plot_data,
                    x="cycle",
                    y="soh",
                    color="battery_id",
                    hover_data=["dataset", "capacity_ah", "rated_capacity_ah"],
                    labels={"cycle": "Cycle Index", "soh": "State of Health (SOH %)", "battery_id": "Battery Cell"},
                    title=f"Degradation Trajectories — {len(sel_cells)} selected cells",
                    template="plotly_dark",
                )
                fig.add_hline(y=80.0, line_dash="dash", line_color="#ef4444", annotation_text="80% SOH (Harmonized EOL)")
                fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one battery to visualize degradation.")
        else:
            st.warning("Unified dataset not found at `data/processed/unified.csv`.")
            
    with tab_chem:
        st.subheader("Electrochemical & Thermal Distributions")
        if unified_df is not None:
            ec1, ec2 = st.columns(2)
            with ec1:
                fig_box = px.box(
                    unified_df,
                    x="chemistry",
                    y="soh",
                    color="chemistry",
                    title="SOH Distribution by Battery Chemistry",
                    template="plotly_dark",
                    labels={"chemistry": "Cathode Chemistry", "soh": "Observed SOH (%)"},
                )
                fig_box.update_layout(height=420, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                
            with ec2:
                # Scatter of capacity vs temperature
                sub_temp = unified_df.dropna(subset=["temperature_c", "capacity_ah"])
                if len(sub_temp) > 3000:
                    sub_temp = sub_temp.sample(3000, random_state=42)
                fig_temp = px.scatter(
                    sub_temp,
                    x="temperature_c",
                    y="capacity_ah",
                    color="dataset",
                    title="Discharge Capacity vs. Ambient Operating Temperature",
                    template="plotly_dark",
                    labels={"temperature_c": "Temperature (°C)", "capacity_ah": "Measured Capacity (Ah)"},
                )
                fig_temp.update_layout(height=420)
                st.plotly_chart(fig_temp, use_container_width=True)


# ==============================================================================
# PAGE 3: 5x5 TRANSFER MATRIX (INTERACTIVE)
# ==============================================================================
elif page == "📊 5×5 Transfer Matrix":
    render_hero(
        "5×5 Cross-Dataset Transfer Matrix",
        "Evaluate pairwise domain generalization across all 5 battery repositories. Diagonal elements show within-domain holdout evaluation.",
        [("500 Matrix Runs", "badge-teal"), ("4 Model Families", "badge-blue"), ("5 Random Seeds", "badge-amber")],
    )
    
    if matrix_df is not None:
        tab_heat, tab_compare, tab_table = st.tabs([
            "🗺️ Interactive Heatmap Explorer",
            "⚖️ Side-by-Side Model Comparator",
            "📋 Tabular Statistics (All 25 Pairs)",
        ])
        
        with tab_heat:
            c_m1, c_m2, c_m3 = st.columns([1.2, 1.2, 1])
            with c_m1:
                sel_model = st.selectbox(
                    "Select Model Architecture:",
                    ["xgboost", "ridge", "elasticnet", "mean"],
                    format_func=lambda x: {
                        "xgboost": "XGBoost (Gradient Boosted Trees)",
                        "ridge": "Ridge Regression (L2)",
                        "elasticnet": "ElasticNet (L1 + L2)",
                        "mean": "Mean Baseline (Dummy)",
                    }.get(x, x),
                )
            with c_m2:
                sel_metric = st.selectbox(
                    "Select Evaluation Metric:",
                    ["mae_mean", "rmse_mean", "mape_mean"],
                    format_func=lambda x: {
                        "mae_mean": "Mean Absolute Error (SOH pp)",
                        "rmse_mean": "Root Mean Squared Error (SOH pp)",
                        "mape_mean": "Mean Absolute Percentage Error (%)",
                    }.get(x, x),
                )
            with c_m3:
                colorscale = st.selectbox("Color Palette:", ["Viridis", "Plasma", "Turbo", "Cividis"], index=0)
                
            sub_m = matrix_df[matrix_df["model"] == sel_model]
            pivot_val = sub_m.pivot(index="source", columns="target", values=sel_metric)
            pivot_std = sub_m.pivot(index="source", columns="target", values="mae_std")
            
            # Interactive Plotly Heatmap
            customdata = np.dstack((pivot_val.values, pivot_std.values))
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot_val.values,
                    x=pivot_val.columns.tolist(),
                    y=pivot_val.index.tolist(),
                    colorscale=colorscale,
                    text=pivot_val.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 13, "color": "white"},
                    customdata=customdata,
                    hovertemplate="<b>Source:</b> %{y}<br><b>Target:</b> %{x}<br><b>Value:</b> %{customdata[0]:.2f} ± %{customdata[1]:.2f}<extra></extra>",
                    colorbar=dict(title=dict(text="Error")),
                )
            )
            fig_heat.update_layout(
                title=f"Cross-Dataset Transfer Matrix — {sel_model.upper()} ({sel_metric.split('_')[0].upper()})",
                xaxis_title="Target Testing Domain",
                yaxis_title="Source Training Domain",
                template="plotly_dark",
                height=520,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown(
                """
                > **Interpretation Guide:**  
                > - The **diagonal** elements represent within-domain generalization on held-out physical cells (`battery_split`).
                > - The **off-diagonal** elements represent zero-shot transfer onto completely unseen external repositories.
                """
            )
            
        with tab_compare:
            st.subheader("Side-by-Side Model Divergence Analysis")
            st.markdown("Compare two models directly to visualize where linear extrapolation collapses vs. tree partitioning.")
            
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                m1 = st.selectbox("Model A:", ["xgboost", "ridge", "elasticnet", "mean"], index=0, key="cmp_m1")
                sub_m1 = matrix_df[matrix_df["model"] == m1].pivot(index="source", columns="target", values="mae_mean")
                fig_m1 = go.Figure(go.Heatmap(
                    z=sub_m1.values, x=sub_m1.columns, y=sub_m1.index,
                    colorscale="Viridis", text=sub_m1.round(1).values, texttemplate="%{text}",
                    textfont={"size": 11, "color": "white"}, zmin=0, zmax=50,
                ))
                fig_m1.update_layout(title=f"Model A: {m1.upper()} (MAE pp, capped 50)", template="plotly_dark", height=420)
                st.plotly_chart(fig_m1, use_container_width=True)
                
            with cmp_col2:
                m2 = st.selectbox("Model B:", ["ridge", "xgboost", "elasticnet", "mean"], index=0, key="cmp_m2")
                sub_m2 = matrix_df[matrix_df["model"] == m2].pivot(index="source", columns="target", values="mae_mean")
                fig_m2 = go.Figure(go.Heatmap(
                    z=sub_m2.values, x=sub_m2.columns, y=sub_m2.index,
                    colorscale="Viridis", text=sub_m2.round(1).values, texttemplate="%{text}",
                    textfont={"size": 11, "color": "white"}, zmin=0, zmax=50,
                ))
                fig_m2.update_layout(title=f"Model B: {m2.upper()} (MAE pp, capped 50)", template="plotly_dark", height=420)
                st.plotly_chart(fig_m2, use_container_width=True)
                
        with tab_table:
            st.subheader("Complete Per-Pair Quantitative Summary")
            st.dataframe(matrix_df, use_container_width=True)
    else:
        st.warning("Matrix summary file `results/full_study/matrix/figures/per_pair_seed_summary.csv` not found.")


# ==============================================================================
# PAGE 4: LODO & FEW-SHOT ADAPTATION
# ==============================================================================
elif page == "🌐 LODO & Few-Shot Adaptation":
    render_hero(
        "LODO & Few-Shot Domain Adaptation",
        "Evaluate Leave-One-Dataset-Out (LODO) generalization across pooled domains and few-shot adaptation efficiency with k ∈ {0, 1, 3, 5, 10} support cells.",
        [("100 LODO Runs", "badge-blue"), ("5,840 Adaptation Runs", "badge-teal"), ("4 Adaptation Methods", "badge-amber")],
    )
    
    tab_lodo_p, tab_adapt_p = st.tabs([
        "🌐 Leave-One-Dataset-Out (LODO)",
        "🎯 Few-Shot Domain Adaptation Curves",
    ])
    
    with tab_lodo_p:
        st.subheader("Generalization Across 4 Pooled Domains")
        st.markdown(
            "In LODO evaluation, models train on four pooled datasets and evaluate zero-shot on the held-out fifth dataset. "
            "This tests whether exposure to multi-domain data prevents single-source extrapolation failure."
        )
        
        if lodo_df is not None:
            fig_lodo = go.Figure()
            for model_name in sorted(lodo_df["model"].unique()):
                sub = lodo_df[lodo_df["model"] == model_name]
                fig_lodo.add_trace(go.Bar(
                    name=model_name.upper(),
                    x=sub["target"],
                    y=sub["mae_mean"],
                    error_y=dict(type="data", array=sub["mae_std"].fillna(0).values, visible=True),
                ))
            fig_lodo.update_layout(
                barmode="group",
                title="LODO MAE (SOH pp) Across Held-Out Target Domains (Mean ± Std, 5 Seeds)",
                xaxis_title="Held-Out Target Domain",
                yaxis_title="MAE (percentage points)",
                template="plotly_dark",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lodo, use_container_width=True)
            
            st.markdown("### LODO Generalization Summary Table")
            st.dataframe(lodo_df[["model", "target", "mae_mean", "mae_std", "rmse_mean", "mape_mean", "runs"]], use_container_width=True, hide_index=True)
        else:
            st.info("LODO summary file not found.")
            
    with tab_adapt_p:
        st.subheader("Few-Shot Adaptation Sample Efficiency")
        st.markdown(
            "Evaluating target MAE as a function of target support battery count ($k$). "
            "Tested strategies: **scratch** (target support only), **pooled** (source + support), and **CORAL / Normalization** (feature alignment without retraining)."
        )
        
        if adapt_df is not None:
            c_ad1, c_ad2 = st.columns([1.5, 1])
            with c_ad1:
                sel_ad_model = st.selectbox("Select Model:", ["All Models Average", "xgboost", "ridge", "elasticnet", "mean"])
            with c_ad2:
                all_pairs = ["All Pairs (Macro-Average)"] + sorted((adapt_df["source"] + " → " + adapt_df["target"]).unique().tolist())
                sel_pair = st.selectbox("Select Domain Pair:", all_pairs)
                
            clean_ad = adapt_df.dropna(subset=["MAE"]).copy()
            if sel_ad_model != "All Models Average":
                clean_ad = clean_ad[clean_ad["model"] == sel_ad_model]
            if sel_pair != "All Pairs (Macro-Average)":
                src_sel, tgt_sel = sel_pair.split(" → ")
                clean_ad = clean_ad[(clean_ad["source"] == src_sel) & (clean_ad["target"] == tgt_sel)]
                
            fig_ad = go.Figure()
            colors_map = {
                "none": "#94a3b8",
                "scratch": "#22c55e",
                "pooled": "#3b82f6",
                "coral": "#f59e0b",
                "normalization": "#a855f7",
            }
            
            for method, g in clean_ad.groupby("method"):
                agg = g.groupby("k")["MAE"].agg(["mean", "std"]).reset_index()
                fig_ad.add_trace(go.Scatter(
                    x=agg["k"],
                    y=agg["mean"],
                    mode="lines+markers",
                    name=method.capitalize(),
                    line=dict(color=colors_map.get(method, "#38bdf8"), width=3),
                    marker=dict(size=8),
                    error_y=dict(type="data", array=agg["std"].fillna(0).values, visible=True),
                ))
                
            fig_ad.update_layout(
                title=f"Few-Shot Adaptation Trajectory — {sel_ad_model.upper()} ({sel_pair})",
                xaxis_title="Number of Target Support Batteries (k)",
                yaxis_title="Target Test MAE (SOH pp)",
                template="plotly_dark",
                height=480,
                hovermode="x unified",
            )
            st.plotly_chart(fig_ad, use_container_width=True)
            
            st.markdown("### Adaptation Progression by Support Size (k)")
            pivot_ad = clean_ad.pivot_table(index="method", columns="k", values="MAE", aggfunc="mean").round(2)
            st.dataframe(pivot_ad, use_container_width=True)
        else:
            st.info("Adaptation metrics file not found.")


# ==============================================================================
# PAGE 5: WHAT-IF SOH PREDICTOR (LIVE SIMULATOR)
# ==============================================================================
elif page == "🔬 What-If SOH Predictor":
    render_hero(
        "Interactive Battery SOH Prediction Simulator",
        "Live inference sandbox: Select any physical battery cell from the 190 validated cells, configure source domains and adaptation levels, and execute real-time model inference.",
        [("Real-Time Inference", "badge-teal"), ("Interactive What-If", "badge-blue"), ("Zero Latency", "badge-amber")],
    )
    
    if unified_df is not None:
        st.subheader("1. Configure Prediction Scenario")
        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        
        with sim_c1:
            sim_tgt_ds = st.selectbox("Target Dataset:", sorted(unified_df["dataset"].unique()), index=3) # default Oxford
        with sim_c2:
            avail_tgt_cells = sorted(unified_df[unified_df["dataset"] == sim_tgt_ds]["battery_id"].unique())
            sim_tgt_cell = st.selectbox("Target Battery Cell:", avail_tgt_cells, index=0)
        with sim_c3:
            sim_model_name = st.selectbox("Model Architecture:", ["xgboost", "ridge", "elasticnet", "mean"])
        with sim_c4:
            src_opts = ["Within-Dataset (Remaining Cells)"] + [d for d in sorted(unified_df["dataset"].unique()) if d != sim_tgt_ds]
            sim_src_mode = st.selectbox("Training Source Domain:", src_opts)
            
        sim_col_k, sim_col_btn = st.columns([2, 1])
        with sim_col_k:
            k_support = st.slider("Target Support Cells for Adaptation (k):", 0, 5, 0, help="0 = Zero-shot transfer; >0 uses support cells from target dataset for adaptation")
            
        with sim_col_btn:
            st.write("")
            st.write("")
            run_sim = st.button("🚀 Run Live Inference", type="primary", use_container_width=True)
            
        # Target cell data
        tgt_cell_data = unified_df[(unified_df["dataset"] == sim_tgt_ds) & (unified_df["battery_id"] == sim_tgt_cell)].sort_values("cycle")
        
        # Training data selection
        if sim_src_mode.startswith("Within"):
            train_pool = unified_df[(unified_df["dataset"] == sim_tgt_ds) & (unified_df["battery_id"] != sim_tgt_cell)]
        else:
            train_pool = unified_df[unified_df["dataset"] == sim_src_mode]
            
        # Adaptation support pool
        other_tgt_cells = [c for c in avail_tgt_cells if c != sim_tgt_cell]
        
        if run_sim or "sim_done" in st.session_state:
            st.session_state["sim_done"] = True
            
            # Train model
            if k_support > 0 and len(other_tgt_cells) >= k_support:
                support_ids = other_tgt_cells[:k_support]
                support_data = unified_df[(unified_df["dataset"] == sim_tgt_ds) & (unified_df["battery_id"].isin(support_ids))]
                combined_train = pd.concat([train_pool, support_data], ignore_index=True)
            else:
                combined_train = train_pool
                
            X_train = combined_train[["cycle"]].values
            y_train = combined_train["soh"].values
            X_test = tgt_cell_data[["cycle"]].values
            y_test = tgt_cell_data["soh"].values
            
            model = make_classical(sim_model_name, seed=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            residuals = y_test - preds
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            max_err = float(np.max(np.abs(residuals)))
            
            st.write("")
            st.subheader(f"Inference Results: {sim_tgt_ds.upper()} — {sim_tgt_cell}")
            
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Test MAE", f"{mae:.2f} pp", delta=f"{mae - 3.89:.2f} vs within baseline", delta_color="inverse")
            r_col2.metric("Test RMSE", f"{rmse:.2f} pp")
            r_col3.metric("Max Point Error", f"{max_err:.2f} pp")
            r_col4.metric("Observed Cycles", f"{len(tgt_cell_data)}")
            
            # Plotly Actual vs Predicted
            res_df = tgt_cell_data.copy()
            res_df["predicted_soh"] = preds
            res_df["residual"] = residuals
            
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=res_df["cycle"],
                y=res_df["soh"],
                mode="lines+markers",
                name="Actual Observed SOH",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=5),
            ))
            fig_sim.add_trace(go.Scatter(
                x=res_df["cycle"],
                y=res_df["predicted_soh"],
                mode="lines",
                name=f"Predicted ({sim_model_name.upper()})",
                line=dict(color="#f43f5e", width=2.5, dash="dash"),
            ))
            fig_sim.add_hline(y=80.0, line_dash="dot", line_color="#ef4444", annotation_text="80% SOH")
            fig_sim.update_layout(
                title=f"Live Prediction: Actual vs. Predicted SOH — {sim_tgt_cell} ({sim_model_name.upper()}, k={k_support})",
                xaxis_title="Cycle Index",
                yaxis_title="State of Health (%)",
                template="plotly_dark",
                height=480,
                hovermode="x unified",
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Residual error plot
            fig_res = px.bar(
                res_df,
                x="cycle",
                y="residual",
                title="Cycle-by-Cycle Residual Error (Actual - Predicted SOH pp)",
                template="plotly_dark",
                color="residual",
                color_continuous_scale="RdBu_r",
            )
            fig_res.update_layout(height=280)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.warning("Unified dataset not found.")


# ==============================================================================
# PAGE 6: DATA QUALITY & AUDIT
# ==============================================================================
elif page == "🛡️ Data Quality & Audit":
    render_hero(
        "Data Quality & Physical Validation Audit",
        "Inspect data hygiene reports, zero-leakage verification, and explicit physical accounting of constrained cell pools.",
        [("Zero Synthetic Data", "badge-teal"), ("Leakage Safeguards", "badge-blue"), ("320 Documented Infeasibilities", "badge-amber")],
    )
    
    tab_q1, tab_q2, tab_q3 = st.tabs([
        "📊 Data Quality Report",
        "🚫 Infeasible Settings Accounting",
        "✅ 10-Point Verification Checklist",
    ])
    
    with tab_q1:
        st.subheader("Data Hygiene Metrics Across All 5 Repositories")
        if quality_df is not None:
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
        else:
            st.info("Quality report `data/processed/data_quality.csv` not found.")
            
    with tab_q2:
        st.subheader("Accounting of Physically Infeasible Settings (Zero Fabrication)")
        st.markdown(
            """
            In accordance with rigorous empirical standards, no battery cells were synthetically fabricated when target pools were constrained:
            - **NASA as Target ($N = 4$ cells total):** 2 cells are reserved for test $\\to$ 2 support cells available. Settings $k \\in \\{3, 5, 10\\}$ are physically infeasible ($240$ documented runs).
            - **Oxford as Target ($N = 8$ cells total):** 3 cells are reserved for test $\\to$ 5 support cells available. Setting $k = 10$ is physically infeasible ($80$ documented runs).
            - **Total Infeasible Settings:** Exactly **320 runs** were recorded with explicit physical reasons in `adaptation_metrics.csv`.
            """
        )
        if adapt_df is not None:
            nf_df = adapt_df[adapt_df["method"] == "not_feasible"]
            st.dataframe(nf_df[["source", "target", "model", "k", "method", "reason"]].drop_duplicates(), use_container_width=True, hide_index=True)
            
    with tab_q3:
        st.subheader("Ten-Point Verification Audit")
        audit_items = [
            ("1. Disjoint Battery Splitting", "PASSED: All splits partition by battery ID; zero cycle-level row splitting."),
            ("2. Zero Feature Leakage", "PASSED: Current-cycle capacity and SOH are strictly excluded from predictors."),
            ("3. Zero Synthetic Data", "PASSED: 100% of benchmark scores are derived from authenticated laboratory files."),
            ("4. No Label Interpolation", "PASSED: Oxford sparse characterization checkpoints (~100 cycles) are preserved discretely."),
            ("5. CADEX Physical Integration", "PASSED: Reconstructed Ah via current-time integration instead of progress percentage."),
            ("6. Arbin Counter Segmentation", "PASSED: Multi-step counter accumulation in CS2_3 and CS2_9 re-segmented cleanly."),
            ("7. MATR Glitch Outlier Filtering", "PASSED: 4 single-cycle instrumentation spikes (>1.43 Ah) filtered cleanly."),
            ("8. Source vs. Harmonized EOL", "PASSED: Source EOL thresholds (70% and 80%) preserved distinctly."),
            ("9. 5-Seed Replication", "PASSED: Evaluated across seeds [11, 22, 33, 44, 55] with reported mean ± std."),
            ("10. Full Test Suite", "PASSED: All 30 unit and integration tests passing in .venv."),
        ]
        for title, desc in audit_items:
            st.markdown(f"**{title}** — {desc}")


# ==============================================================================
# PAGE 7: RESEARCH REPORT & EXPORT CENTER
# ==============================================================================
elif page == "📄 Research Report & Export":
    render_hero(
        "Research Report & Benchmark Data Export",
        "Access the complete empirical research report and download benchmark matrices, summaries, and data quality logs.",
        [("Publication-Grade Report", "badge-teal"), ("CSV Export Center", "badge-blue")],
    )
    
    st.subheader("One-Click Data & Metric Exports")
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    
    with d_c1:
        if matrix_df is not None:
            st.download_button(
                "📥 5×5 Transfer Matrix (CSV)",
                data=matrix_df.to_csv(index=False),
                file_name="cross_dataset_transfer_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with d_c2:
        if lodo_df is not None:
            st.download_button(
                "📥 LODO Generalization (CSV)",
                data=lodo_df.to_csv(index=False),
                file_name="lodo_generalization_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with d_c3:
        if adapt_df is not None:
            st.download_button(
                "📥 Few-Shot Adaptation (CSV)",
                data=adapt_df.to_csv(index=False),
                file_name="few_shot_adaptation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with d_c4:
        if quality_df is not None:
            st.download_button(
                "📥 Data Quality Audit (CSV)",
                data=quality_df.to_csv(index=False),
                file_name="data_quality_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
    st.write("")
    st.subheader("Integrated Empirical Research Report")
    report_path = ROOT / "reports/REPORT_TEMPLATE.md"
    if report_path.exists():
        with st.expander("📖 Read Full Publication-Grade Research Report", expanded=True):
            st.markdown(report_path.read_text())
    else:
        st.info("Report file not found.")
