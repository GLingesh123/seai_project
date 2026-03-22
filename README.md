# PROJECT TITLE: Adaptive Learning in Non-Stationary Computer Science Environments with Self-Evolving AI Systems ⚡

### ABSTRACT
Many real-world Computer Science and Engineering (CSE) applications such as cybersecurity monitoring, IoT telemetry, and software log streams operate in non-stationary environments where data distributions change over time. Such distribution shifts, known as concept drift, significantly degrade the performance of static machine learning models. This work presents a Self-Evolving Artificial Intelligence (SEAI) framework for streaming environments that integrates synthetic drift simulation, online drift detection, and adaptive continual learning. The system generates controlled sudden and gradual drift scenarios using a configurable synthetic stream generator and applies online drift detection based on error signal monitoring. Upon drift detection, the model adapts using experience replay and Elastic Weight Consolidation (EWC) to reduce catastrophic forgetting while preserving prior knowledge. A modular adaptation loop coordinates streaming, detection, retraining, and evaluation, supported by automated logging and visualization tools. Experimental results on controlled drift streams demonstrate improved post-drift recovery and stability compared to static baselines, validating the effectiveness of the SEAI approach for adaptive learning in non-stationary data streams.

---

Welcome to the SEAI repository! This README serves as a master execution guide to seamlessly clone, configure, and execute the complete pipeline sequence locally.

---

## 🛠️ Step 1: Project Setup & Installation

Follow these exact steps to set up your local development environment without breaking system dependencies on your machine.

### 1. Clone the Repository
Open your terminal (PowerShell, CMD, or Bash) and clone this exact codebase:
```bash
git clone <repository_url_here>
cd seai_project
```

### 2. Create the Virtual Environment (Crucial)
Never manually install project dependencies globally on your PC. Let's create an isolated Python virtual environment (`.venv`) so the SEAI modules stay locked inside this folder.
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment
Activate the environment so your terminal knows to use the isolated SEAI workspace.
**For Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate
```
*(You should see `(.venv)` structurally prefixing your command line prompt now).*

### 4. Install Core Packages (Requirements)
Install all the underlying mathematics engines (PyTorch, Pytest, Pandas, Streamlit).
```bash
pip install -r requirements.txt
```

---

## 🚀 Step 2: Running the Graphical Dashboard (UI)

We developed an incredible natively-hosted local web dashboard to execute the data drift scenarios and track the live memory retention graphs dynamically.

To boot up the complete graphical interface, execute this command directly from the `seai_project` root folder:
```bash
streamlit run ui/streamlit_app.py
```
*Your default browser should automatically populate and route to `http://localhost:8501`. From the sidebar, you can manually select the Concept Drift scenario (Gradual, Sudden, Recurring) and click **Execute Pipeline** to watch the mathematical metrics cleanly render in real-time!*

---

## 💻 Step 3: Running the Backend Synthetics (CLI Options)

If you don't want to use the Web UI and simply want to generate the pure `.csv` mathematics matrices directly from the terminal, you can run the monolithic backend core script natively:

```bash
# Run a Gradual anomaly sequence for exactly 200 pipeline frames
python main.py --scenario gradual --steps 200

# Run a Violent/Sudden anomaly sequence
python main.py --scenario sudden --steps 500

# Run a Repeating anomalies sequence
python main.py --scenario recurring --steps 300
```
*The pure structural output metrics will physically manifest internally into `logs/csv/` and the generated architectural image proofs will populate inside `results/baseline_vs_seai_comparisons/`!*

---

## 🧪 Step 4: Academic Unit Testing Guide

Because this is a structurally complex Artificial Intelligence project, we wrote strict **Unit Tests** to mathematically isolate and definitively prove every single computational concept (e.g. Memory caching, EWC derivatives, Active neural matrices). 

If you make *any* changes to the core code, you **MUST** run these tests to ensure you haven't accidentally broken the overall mathematical algorithms.

### Run the Entire Testing Battery at Once (Recommended)
This command will sweep the entire directory and execute all 12 logic validations simultaneously:
```bash
pytest tests/ -v
```

### Executing Specific Component Tests Individually
If a specific architecture component crashes, you can isolate and test it directly using its script:

**Test the Concept Drift Detection Algorithms (ADWIN / PageHinkley checks):**
```bash
pytest tests/test_drift_manager.py
pytest tests/test_drift.py
```

**Test the Catastrophic Forgetting Defenses (EWC Mathematics):**
```bash
pytest tests/test_ewc.py
pytest tests/test_forgetting_ewc.py
```

**Test the Primary Neural Network Layer Engine (MLP Architecture):**
```bash
pytest tests/test_mlp.py
```

**Test the High-Speed Memory Matrix Cache (Retention check):**
```bash
pytest tests/test_replay.py
```

**Test the Synthetic Stream Flow Generation Logic:**
```bash
pytest tests/test_stream.py
pytest tests/test_stream_loader.py
```

**Test the Core Engine Execution Architecture Loops:**
```bash
pytest tests/test_adaptation_loop.py
pytest tests/test_trainer.py
pytest tests/test_logger.py
```

---

*If you need specific speaking notes for the project defense, please refer directly to the `IMPLEMENTATION_GUIDE.md` blueprint!*
