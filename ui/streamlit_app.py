import streamlit as st
import subprocess
from pathlib import Path

# -------------------------------------------------
# Config
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

st.set_page_config(
    page_title="SEAI Module Runner",
    layout="centered"
)

st.title("SEAI Experiment Runner UI")
st.write("Select a module → run experiment → view graph.")

# -------------------------------------------------
# Module Map
# ready = implemented
# -------------------------------------------------

MODULES = {
    "Drift Detection — SEAI vs Baseline": {
        "cmd": ["python", "-m", "experiments.seai_vs_baseline_drift_plot"],
        "plot": "seai_vs_baseline_drift.png",
        "ready": True
    },

    "Forgetting — Replay + EWC vs Baseline": {
        "cmd": None,
        "plot": None,
        "ready": False
    },

    "Adaptation Latency": {
        "cmd": None,
        "plot": None,
        "ready": False
    },

    "Method Comparison (Baseline / Replay / SEAI)": {
        "cmd": None,
        "plot": None,
        "ready": False
    },
}

# -------------------------------------------------
# UI
# -------------------------------------------------

choice = st.selectbox("Choose Module", list(MODULES.keys()))
info = MODULES[choice]

if not info["ready"]:
    st.warning("🚧 This module is still under development.")
    st.stop()

run_btn = st.button("Run Selected Module")

# -------------------------------------------------
# Runner
# -------------------------------------------------

if run_btn:

    cmd = info["cmd"]
    plot_name = info["plot"]

    st.info("Running experiment... please wait.")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        st.text_area(
            "Execution Log",
            result.stdout + "\n" + result.stderr,
            height=300
        )

    except Exception as e:
        st.error(f"Execution failed: {e}")
        st.stop()

    # ---------- plot display ----------

    plot_path = PLOTS_DIR / plot_name

    if plot_path.exists():
        st.success("Plot generated successfully.")
        st.image(str(plot_path))
    else:
        st.error(f"Expected plot not found:\n{plot_path}")

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")
st.caption("SEAI — Self Evolving AI Experiments UI")
