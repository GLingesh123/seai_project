import streamlit as st
import subprocess
import sys
import os
import time
from pathlib import Path
from PIL import Image
import re
import pandas as pd
import json
import base64

# =====================================================
# BACKGROUND OVERRIDE (BASE64 INJECTION)
# =====================================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# =====================================================
# UI / CSS CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="SEAI | Continual Learning Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Base64 Background
PROJECT_ROOT = Path(__file__).resolve().parent.parent
bg_path = PROJECT_ROOT / "ui/assets/bg.png"

if bg_path.exists():
    bg_b64 = get_base64_of_bin_file(bg_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(10,15,30,0.85), rgba(10,15,30,0.95)), url("data:image/png;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background: #0b101e; }</style>", unsafe_allow_html=True)

st.markdown("""
<style>
    :root {
        --glass-bg: rgba(15, 25, 40, 0.65);
        --accent: #00f2fe;
        --secondary: #4facfe;
    }
    
    /* Terminal Console Styling */
    div[data-testid="stCodeBlock"] {
        background-color: rgba(5, 5, 10, 0.95) !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        border-radius: 8px;
    }
    div[data-testid="stCodeBlock"] * {
        color: #00ff00 !important;
        font-family: monospace;
    }
    
    /* Glassmorphic Metric Panels */
    div[data-testid="stMetricValue"] {
        font-size: 34px !important;
        color: var(--accent) !important;
        text-shadow: 0px 0px 20px rgba(0, 242, 254, 0.4);
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] {
        background: var(--glass-bg);
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: transform 0.3s ease, border 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 242, 254, 0.6);
        box-shadow: 0 15px 45px rgba(0, 242, 254, 0.15);
    /* Glowing Emojis */
    .glowing-emoji {
        filter: drop-shadow(0px 0px 8px #00f2fe) drop-shadow(0px 0px 18px #4facfe) hue-rotate(170deg) saturate(400%);
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================

st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 25px; padding-top: 10px;'>
    <h1 style='font-size: 3rem; margin-bottom: 0px;'>SEAI <span class='glowing-emoji'>⚡</span></h1>
    <p style='color: #00f2fe; font-weight: 700; letter-spacing: 3px; font-size: 0.85rem; margin-top: -5px;'>CONTINUAL LEARNING</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

with st.sidebar.container(border=True):
    st.markdown("<h3 style='color: #ffffff; font-weight: 700; margin-bottom: 15px;'><span class='glowing-emoji'>⚙️</span> Settings</h3>", unsafe_allow_html=True)
    scenario = st.selectbox("Drift Type", ["gradual", "sudden", "recurring", "none"], index=0, help="Controls the internal severity and structure of the concept data shift.")
    steps = st.slider("Steps", min_value=50, max_value=800, value=200, step=50, help="Total execution length determining hardware benchmarking limits.")

st.sidebar.divider()

with st.sidebar.container(border=True):
    st.markdown("<h3 style='color: #ffffff; font-weight: 700; margin-bottom: 15px;'><span class='glowing-emoji'>🧬</span> EWC Configuration</h3>", unsafe_allow_html=True)
    ewc_lambda = st.slider("EWC Lambda", min_value=100.0, max_value=30000.0, value=20000.0, step=500.0, help="Math scale protecting fundamental architecture elasticity during continuous learning loops.")

st.sidebar.divider()

st.sidebar.info("**Ready.** Configure settings and execute pipeline.")

st.sidebar.markdown("<br>", unsafe_allow_html=True)
run_clicked = st.sidebar.button("Execute Pipeline", use_container_width=True)

# =====================================================
# MAIN DASHBOARD / TABS
# =====================================================
st.markdown("<h1>SEAI Dashboard <span class='glowing-emoji'>🧠</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #cbd5e1 !important;'>Continual learning model adaptation and metrics.</p>", unsafe_allow_html=True)

tab_dash, tab_console, tab_artifacts = st.tabs([
    "  Dashboard  ", 
    "  Terminal  ", 
    "  Artifacts  "
])

# =====================================================
# SESSION STATE INIT
# =====================================================
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""

if run_clicked:
    st.session_state.has_run = True
    
    with tab_dash:
        st.info("Injecting modified limits into backend configurations...")
        config_path = PROJECT_ROOT / "config.py"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_code = f.read()
            new_config = re.sub(r'EWC_LAMBDA\s*=\s*[\d.]+', f'EWC_LAMBDA = {ewc_lambda}', config_code)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(new_config)
        except Exception as e:
            st.warning("Could not patch config dynamically. Using defaults.")
            
        progress_bar = st.progress(5)
        
    with tab_console:
        st.markdown("###  Backend Execution Vector")
        log_container = st.empty()
        
        process = subprocess.Popen(
            [sys.executable, "main.py", "--scenario", scenario, "--steps", str(steps)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(PROJECT_ROOT)
        )
        
        live_log = ""
        report = ""
        hide_line = False
        for line in iter(process.stdout.readline, ''):
            if "___HIDDEN_REPORT_START___" in line:
                hide_line = True
                continue
            if "___HIDDEN_REPORT_END___" in line:
                hide_line = False
                continue
                
            if hide_line:
                report += line
            else:
                live_log += line
                log_container.code(live_log, language='bash')
        
        process.stdout.close()
        process.wait()
        
        st.session_state.terminal_log = live_log
        st.session_state.final_report = report
        
    with tab_dash:
        progress_bar.progress(100)
        if process.returncode != 0:
            st.error("Execution Terminated with Errors. View Live Terminal Stream tab for details.")
            st.session_state.has_run = False
        else:
            st.success("✓ Pipeline sequence finalized mathematically!")

# =====================================================
# TAB 1: RENDERING LOGIC
# =====================================================
with tab_dash:
    if st.session_state.has_run:
        report = st.session_state.final_report
        if len(report) > 5:
            st.divider()
            st.subheader("Architectural Verification Metrics")
            
            try:
                a_base = float(re.search(r'A_BASE:\s+([0-9.]+)', report).group(1))
                a_seai = float(re.search(r'A_SEAI:\s+([0-9.]+)', report).group(1))
                t_base = float(re.search(r'T_BASE:\s+([0-9.]+)', report).group(1))
                t_seai = float(re.search(r'T_SEAI:\s+([0-9.]+)', report).group(1))
                m_base = float(re.search(r'M_BASE:\s+([0-9.]+)', report).group(1))
                m_seai = float(re.search(r'M_SEAI:\s+([0-9.]+)', report).group(1))
                f_base = float(re.search(r'F_BASE:\s+([0-9.]+)', report).group(1))
                f_seai = float(re.search(r'F_SEAI:\s+([0-9.]+)', report).group(1))

                st.markdown("##### Telemetry Dashboard")
                c1, c2, c3, c4 = st.columns(4)
                
                c1.metric("Drift Accuracy", f"{a_seai:.2%}", f"{(a_seai - a_base):+.2%} over Baseline")
                c2.metric("Forgetting Rate", f"{f_seai:.2%}", f"{(f_seai - f_base):+.2%} vs Baseline", delta_color="inverse")
                c3.metric("Adaptation Time", f"{t_seai:.1f}ms", f"Optimal <25ms compute bounds", delta_color="normal")
                c4.metric("Memory Usage", f"{m_seai:.1f}KB", f"Strict <20KB memory limit", delta_color="normal")
                
            except Exception as e:
                st.info("Advanced Results Compiled:")
                st.text(report)
            
            st.markdown("---")
            
            st.subheader("Targeted Hardware Architecture Analytics")
            st.markdown("Direct evaluation between rigid Baseline bounds and the SEAI continuous adaptation engine.")
            
            try:
                st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/accuracy_comparison.jpg"), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/drift_accuracy.jpg"), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/accuracy_advantage.jpg"), use_container_width=True)
                
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/loss_comparison.jpg"), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/forgetting_comparison.jpg"), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/latency_comparison.jpg"), use_container_width=True)
                with gc2:
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/forgetting_rate.jpg"), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/memory_usage.jpg"), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image(str(PROJECT_ROOT / "results/baseline_vs_seai_comparisons/adaptation_time.jpg"), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Waiting for telemetry render streams... ({e})")
    else:
        st.info("👈 Please start the framework by clicking **Execute Pipeline** in the sidebar to hydrate the dashboard.")

# =====================================================
# TAB 2: LIVE CONSOLE
# =====================================================
with tab_console:
    if not run_clicked:
        if st.session_state.has_run:
            st.markdown("###  Backend Execution Vector")
            st.code(st.session_state.terminal_log, language='bash')
        else:
            st.text("Terminal idle. Execute a sequence to monitor mathematical vectors in real time.")

# =====================================================
# TAB 3: RAW ARTIFACTS
# =====================================================
with tab_artifacts:
    st.subheader("Deep Data Explorer")
    st.markdown("Inspect granular CSV matrices generated natively by the adaptation execution loops.")
    
    model_type = st.radio("Select Architecture Type", ["SEAI (Adaptive)", "Baseline (Rigid)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    csv_dir = PROJECT_ROOT / "logs/csv"
    if csv_dir.exists():
        target_prefix = "seai_test_run" if "SEAI" in model_type else "baseline_test_run"
        files = [f for f in sorted(csv_dir.glob("*.csv")) if f.name.startswith(target_prefix)]
        
        if files:
            f = st.selectbox("Mount Core Artifact", files, format_func=lambda p: p.name)
            df = pd.read_csv(f)
            st.dataframe(df, use_container_width=True)
            
            with open(f, "rb") as fp:
                st.download_button("Export CSV Matrix", fp, file_name=f.name)
        else:
            st.info(f"No {model_type} logs populated yet. Run the simulation first.")
    else:
        st.warning("Evaluation directory missing. Run sequence first.")
