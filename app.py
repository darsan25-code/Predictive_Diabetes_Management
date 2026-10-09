"""
Local Results Viewer for Diabetes Digital Twin Project (Streamlit App).

Persistent Invariant (AGENTS.md Rule 7):
  "Research prototype. Not a medical device. Not for clinical decisions."
  Zero dosing or insulin recommendations provided.
"""
from __future__ import annotations

import glob
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent

# Set page configuration
st.set_page_config(
    page_title="T1D Digital Twin — Results Viewer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mandatory persistent clinical disclaimer banner
DISCLAIMER_TEXT = "⚠️ **Research prototype. Not a medical device. Not for clinical decisions.**"
st.warning(DISCLAIMER_TEXT)


@st.cache_data
def load_config() -> dict:
    """Load configuration from YAML."""
    cfg_path = REPO_ROOT / "configs" / "data_default.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


cfg = load_config()

# Sidebar Navigation
st.sidebar.title("🩺 T1D Digital Twin")
st.sidebar.markdown(f"*{DISCLAIMER_TEXT}*")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Results & Experiments", "Synthetic Patients", "Simulator Cohorts (UVA/Padova)"],
)


# ==============================================================================
# Page 1: Overview
# ==============================================================================
if page == "Overview":
    st.header("📊 Project Status & Phase Tracker")
    st.caption("Data Origin: Verified Research Artifacts & Live Test Manifests")

    status_file = REPO_ROOT / "STATUS.md"
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            status_text = f.read()
        st.markdown(status_text)
    else:
        st.error("STATUS.md Not available.")


# ==============================================================================
# Page 2: Results & Experiments
# ==============================================================================
elif page == "Results & Experiments":
    st.header("🔬 Experiment Artifacts & Reports")
    st.caption("Data Origin: Verified Simulation & Parameter Estimation Experiments")

    exp_dir = REPO_ROOT / "experiments"
    experiments_list = [
        {
            "phase": "Phase 2: Bergman Minimal Model Parameter Estimation",
            "plot": "phase2_fit.png",
            "report": "phase2_bergman_fit.txt",
            "origin": "synthetic",
        },
        {
            "phase": "Phase 3: Parameter Calibration & Identifiability Analysis",
            "plot": "phase3_identifiability.png",
            "report": "phase3_identifiability_report.md",
            "origin": "synthetic",
        },
        {
            "phase": "Phase 4: Hybrid Residual Model Multi-Horizon Forecasting",
            "plot": "phase4_horizons.png",
            "report": "phase4_metrics_report.md",
            "origin": "estimated",
        },
        {
            "phase": "Phase 5: State Estimation via Extended Kalman Filter (EKF)",
            "plot": "phase5_kalman.png",
            "report": "phase5_validation_report.md",
            "origin": "estimated",
        },
        {
            "phase": "Phase 6: SimGlucose (UVA/Padova) Independent Simulator Validation",
            "plot": "phase6_cohort_match.png",
            "report": "phase6_validation_report.md",
            "origin": "simulated",
        },
        {
            "phase": "Phase 7: Receding-Horizon Model Predictive Controller (MPC)",
            "plot": "phase7_mpc.png",
            "report": "phase7_mpc_report.md",
            "origin": "simulated",
        },
    ]

    for exp in experiments_list:
        with st.expander(f"📁 {exp['phase']} (Data Origin: `{exp['origin']}`)", expanded=True):
            col1, col2 = st.columns([1.2, 1.0])

            # Left Column: Plot
            plot_path = exp_dir / exp["plot"]
            with col1:
                st.subheader("Visualization")
                if plot_path.exists():
                    st.image(str(plot_path), use_column_width=True, caption=f"{exp['plot']} (Units: mg/dL, Origin: {exp['origin']})")
                else:
                    st.info(f"Plot `{exp['plot']}`: Not available")

            # Right Column: Report
            report_path = exp_dir / exp["report"]
            with col2:
                st.subheader("Performance Report")
                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        rep_text = f.read()
                    st.markdown(rep_text)
                else:
                    st.info(f"Report `{exp['report']}`: Not available")


# ==============================================================================
# Page 3: Synthetic Patients
# ==============================================================================
elif page == "Synthetic Patients":
    st.header("👤 Synthetic Patient Cohort Traces")
    st.markdown("**Data Origin: `synthetic`** *(Physiological simulation with unmodeled effects. Not real patient data).*")

    processed_dir = REPO_ROOT / "data" / "processed"
    parquet_files = sorted(glob.glob(str(processed_dir / "patient_*.parquet")))

    if not parquet_files:
        st.info("Processed patient files in `data/processed/`: Not available. Run data pipeline to generate.")
    else:
        patient_options = [Path(p).stem.replace("patient_", "") for p in parquet_files]
        selected_patient = st.selectbox("Select Patient Trace", patient_options)

        selected_file = processed_dir / f"patient_{selected_patient}.parquet"
        if selected_file.exists():
            df_patient = pd.read_parquet(selected_file)
            st.write(f"**Patient**: `{selected_patient}` | **Readings**: `{len(df_patient)}` | **Data Origin**: `synthetic` | **Units**: `mg/dL`")

            # Chart
            chart_df = df_patient.set_index("timestamp")[["glucose_mgdL"]]
            st.line_chart(chart_df, y="glucose_mgdL")

            # Summary metrics
            g_vals = df_patient["glucose_mgdL"].values
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Glucose", f"{np.mean(g_vals):.1f} mg/dL")
            col2.metric("Time in Range (70-180)", f"{np.mean((g_vals >= 70) & (g_vals <= 180)) * 100:.1f}%")
            col3.metric("Time Below Range (<70)", f"{np.mean(g_vals < 70) * 100:.1f}%")
            col4.metric("Time Above Range (>180)", f"{np.mean(g_vals > 180) * 100:.1f}%")

            with st.expander("View Raw Data"):
                st.dataframe(df_patient.head(100))
        else:
            st.info(f"Patient file `{selected_file.name}`: Not available")


# ==============================================================================
# Page 4: Simulator Cohorts (UVA/Padova)
# ==============================================================================
elif page == "Simulator Cohorts (UVA/Padova)":
    st.header("🧪 UVA/Padova Simulator Benchmark Cohorts")
    st.markdown("**Data Origin: `simulated (UVA/Padova - simglucose)`** *(Open-source physiological benchmark. Not real patient data).*")

    from src.data.loaders import SimGlucoseLoader
    from src.data.preprocessor import Preprocessor

    cohort_choice = st.selectbox("Select Age Cohort", ["Adult", "Adolescent", "Child"])
    patient_map = {
        "Adult": "adult#001",
        "Adolescent": "adolescent#001",
        "Child": "child#001",
    }
    target_patient = patient_map[cohort_choice]

    st.write(f"Displaying 24-hour benchmark trajectory for `{target_patient}` (Units: `mg/dL`, Data Origin: `simulated`).")

    @st.cache_data
    def get_sim_trace(p_name: str) -> pd.DataFrame:
        loader = SimGlucoseLoader(seed=42)
        raw_df = loader.load_patient(p_name, duration_hours=24.0, seed=42)
        preproc = Preprocessor(dt_minutes=5.0)
        return preproc.transform(raw_df)

    try:
        df_sim = get_sim_trace(target_patient)
        chart_sim = df_sim.set_index("timestamp")[["glucose_mgdL"]]
        st.line_chart(chart_sim, y="glucose_mgdL")

        g_sim = df_sim["glucose_mgdL"].values
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Glucose", f"{np.mean(g_sim):.1f} mg/dL")
        col2.metric("TIR [70, 180]", f"{np.mean((g_sim >= 70) & (g_sim <= 180)) * 100:.1f}%")
        col3.metric("TBR (<70)", f"{np.mean(g_sim < 70) * 100:.1f}%")
        col4.metric("TAR (>180)", f"{np.mean(g_sim > 180) * 100:.1f}%")
    except Exception as e:
        st.info(f"Simulator trace for {target_patient}: Not available ({e})")
