import streamlit as st
import subprocess
from pathlib import Path
import pandas as pd
import json
import time

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
CSV_DIR = PROJECT_ROOT / "results" / "csv"
JSON_DIR = PROJECT_ROOT / "results" / "json"

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="SEAI Drift Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS for professional look
# -------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1E3A8A;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #2563EB;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2563EB 0%, #1E3A8A 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
        transform: translateY(-1px);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F9FAFB;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4B5563;
        font-weight: 500;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        color: #1E3A8A;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Footer */
    footer {
        text-align: center;
        color: #6B7280;
        padding: 2rem 0;
    }
    
    /* DataFrame */
    .dataframe {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header Section
# -------------------------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/experiment.png", width=80)  # placeholder icon
with col2:
    st.title("SEAI Drift Lab")
    st.markdown("#### *Streamlined Experimentation for Concept Drift Detection*")

st.markdown("---")

# -------------------------------------------------
# Module Map
# -------------------------------------------------
MODULES = {
    "Drift Detection — SEAI vs Baseline": {
        "cmd": ["python", "-m", "experiments.seai_vs_baseline_drift_plot"],
        "plot": "seai_vs_baseline_drift.png",
        "ready": True,
        "description": "Compare SEAI drift detection against baseline methods."
    },
    "Forgetting — Replay + EWC": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "description": "Analyze forgetting in continual learning with replay and EWC."
    },
    "Adaptation Latency": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "description": "Measure how quickly models adapt to new data distributions."
    },
    "Method Comparison": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "description": "Comprehensive comparison of all drift adaptation methods."
    },
}

# -------------------------------------------------
# Main Area - Experiment Runner
# -------------------------------------------------
with st.container():
    st.header("🚀 Experiment Runner")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        choice = st.selectbox(
            "Select Module",
            list(MODULES.keys()),
            help="Choose the experiment module you want to run."
        )
        info = MODULES[choice]
        st.caption(info["description"])
    
    with col_right:
        st.write("")  # vertical spacer
        st.write("")
        if info["ready"]:
            run_clicked = st.button("▶ Run Experiment", use_container_width=True)
        else:
            st.button("⏳ Under Development", disabled=True, use_container_width=True)

if info["ready"] and run_clicked:
    with st.spinner("Running experiment... this may take a few seconds."):
        try:
            result = subprocess.run(
                info["cmd"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            
            # Show logs in an expander
            with st.expander("📋 Execution Log", expanded=False):
                st.text_area(
                    "Console Output",
                    result.stdout + "\n" + result.stderr,
                    height=200,
                    label_visibility="collapsed"
                )
            
            # Check for plot
            plot_path = PLOTS_DIR / info["plot"]
            if plot_path.exists():
                st.success("✅ Plot generated successfully!")
                st.image(str(plot_path), use_container_width=True)
            else:
                st.warning(f"⚠️ Plot file not found at: {plot_path}")
                
        except Exception as e:
            st.error(f"❌ Execution failed: {str(e)}")

st.markdown("---")

# -------------------------------------------------
# Logs Viewer - Two column layout for better space usage
# -------------------------------------------------
st.header("📊 Experiment Artifacts")

col_csv, col_json = st.columns(2)

# =========================
# CSV Column
# =========================
with col_csv:
    st.subheader("📁 CSV Results")
    if not CSV_DIR.exists():
        st.warning("CSV folder not found.")
    else:
        csv_files = sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files:
            st.info("No CSV files available.")
        else:
            csv_file = st.selectbox(
                "Select file",
                csv_files,
                format_func=lambda p: p.name,
                key="csv_select"
            )
            
            action = st.radio(
                "Action",
                ["View (first 500 rows)", "Download"],
                horizontal=True,
                key="csv_action"
            )
            
            if action.startswith("View"):
                df = pd.read_csv(csv_file)
                st.dataframe(df.head(500), use_container_width=True)
            else:
                with open(csv_file, "rb") as f:
                    st.download_button(
                        "📥 Download CSV",
                        f,
                        file_name=csv_file.name,
                        mime="text/csv",
                        use_container_width=True
                    )

# =========================
# JSON Column
# =========================
with col_json:
    st.subheader("📋 JSON Results")
    if not JSON_DIR.exists():
        st.warning("JSON folder not found.")
    else:
        json_files = sorted(JSON_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            st.info("No JSON files available.")
        else:
            json_file = st.selectbox(
                "Select file",
                json_files,
                format_func=lambda p: p.name,
                key="json_select"
            )
            
            action = st.radio(
                "Action",
                ["View", "Download"],
                horizontal=True,
                key="json_action"
            )
            
            if action == "View":
                with open(json_file, "r") as f:
                    data = json.load(f)
                st.json(data, expanded=1)
            else:
                with open(json_file, "rb") as f:
                    st.download_button(
                        "📥 Download JSON",
                        f,
                        file_name=json_file.name,
                        mime="application/json",
                        use_container_width=True
                    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<footer>SEAI — Drift Detection Experiment UI | v1.0.0</footer>",
    unsafe_allow_html=True
)