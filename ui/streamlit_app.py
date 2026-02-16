import streamlit as st
import subprocess
from pathlib import Path
import pandas as pd
import json

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
    page_title="SEAI Adaptive Learning Workbench",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# Clean Professional Style
# -------------------------------------------------
st.markdown("""
<style>
h1 {color:#1F2937; font-weight:700;}
h2 {color:#2563EB; font-weight:600;}
.stButton > button {
    background:#2563EB;
    color:white;
    border-radius:8px;
    font-weight:600;
}
.stButton > button:hover {
    background:#1E40AF;
}
.block {
    padding:16px;
    border:1px solid #E5E7EB;
    border-radius:10px;
    background:#F9FAFB;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("SEAI Adaptive Learning Workbench")

st.markdown("""
A modular experiment platform for evaluating **stream learning**,  
**concept drift handling**, and **continual learning strategies**.

Use this interface to:
- run experiment modules
- generate evaluation plots
- inspect CSV metrics
- download JSON experiment logs
""")

st.markdown("---")

# -------------------------------------------------
# Modules
# -------------------------------------------------
MODULES = {
    "SEAI vs Static Model — Drift Handling": {
        "cmd": ["python", "-m", "experiments.seai_vs_baseline_drift_plot"],
        "plot": "seai_vs_baseline_drift.png",
        "ready": True,
        "purpose": "Evaluates adaptive learning vs frozen model under distribution shift."
    },

    "Continual Learning — Forgetting Study": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "purpose": "Measures knowledge retention with replay and regularization."
    },

    "Adaptation Latency Analysis": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "purpose": "Quantifies recovery speed after drift detection."
    },

    "Method Benchmark Suite": {
        "cmd": None,
        "plot": None,
        "ready": False,
        "purpose": "Compares multiple adaptation strategies across scenarios."
    },
}

# -------------------------------------------------
# Runner Section
# -------------------------------------------------
st.header("Experiment Runner")

left, right = st.columns([3,1])

with left:
    choice = st.selectbox("Select Experiment Module", list(MODULES.keys()))
    info = MODULES[choice]

    st.markdown(f"""
<div class="block">
<b>Module Purpose</b><br>
{info["purpose"]}
</div>
""", unsafe_allow_html=True)

with right:
    st.write("")
    if info["ready"]:
        run_clicked = st.button("Run Module", use_container_width=True)
    else:
        st.button("Module Not Ready", disabled=True, use_container_width=True)

# -------------------------------------------------
# Run Experiment
# -------------------------------------------------
if info["ready"] and run_clicked:

    with st.spinner("Executing experiment pipeline..."):
        try:
            result = subprocess.run(
                info["cmd"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )

            with st.expander("Execution Log"):
                st.text(result.stdout + "\n" + result.stderr)

            plot_path = PLOTS_DIR / info["plot"]

            if plot_path.exists():
                st.success("Plot generated successfully")
                st.image(str(plot_path), use_container_width=True)
            else:
                st.warning("Expected plot file not found")

        except Exception as e:
            st.error(str(e))

st.markdown("---")

# -------------------------------------------------
# Artifact Browser
# -------------------------------------------------
st.header("Experiment Artifacts")

st.markdown("""
Browse generated experiment outputs.  
CSV = step metrics  
JSON = structured run logs  
""")

col_csv, col_json = st.columns(2)

# ---------------- CSV ----------------
with col_csv:
    st.subheader("CSV Metrics")

    if CSV_DIR.exists():
        files = sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

        if files:
            f = st.selectbox("Select CSV", files, format_func=lambda p: p.name)

            action = st.selectbox("Action", ["Preview", "Download"], key="csv_action")

            if action == "Preview":
                df = pd.read_csv(f)
                st.dataframe(df.head(500), use_container_width=True)
            else:
                with open(f, "rb") as fp:
                    st.download_button("Download CSV", fp, file_name=f.name)

        else:
            st.info("No CSV files yet")
    else:
        st.warning("CSV directory missing")

# ---------------- JSON ----------------
with col_json:
    st.subheader("JSON Logs")

    if JSON_DIR.exists():
        files = sorted(JSON_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if files:
            f = st.selectbox("Select JSON", files, format_func=lambda p: p.name)

            action = st.selectbox("Action", ["Preview", "Download"], key="json_action")

            if action == "Preview":
                with open(f) as fp:
                    st.json(json.load(fp))
            else:
                with open(f, "rb") as fp:
                    st.download_button("Download JSON", fp, file_name=f.name)

        else:
            st.info("No JSON logs yet")
    else:
        st.warning("JSON directory missing")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("SEAI — Self Evolving Adaptive Intelligence | Experiment Platform")
