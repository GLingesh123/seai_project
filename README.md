# Adaptive Learning in Non-Stationary Computer Science Environments with Self-Evolving AI Systems

Many real-world Computer Science and Engineering (CSE) applications such as cybersecurity monitoring, IoT telemetry, and software log streams operate in non-stationary environments where data distributions change over time. Such distribution shifts, known as **concept drift**, significantly degrade the performance of static machine learning models. 

This project presents a **Self-Evolving Artificial Intelligence (SEAI)** framework for streaming environments. It integrates synthetic drift simulation, online drift detection, and adaptive continual learning. The system generates controlled sudden and gradual drift scenarios, applies online detection, and subsequently adapts using Experience Replay and Elastic Weight Consolidation (EWC) to preserve prior knowledge without catastrophic forgetting.

---

##  Core Concepts & How It Works

### Concept Drift
In real-world streams, the statistical properties of targeted variables evolve. SEAI utilizes the **ADWIN (Adaptive Windowing)** algorithm via a central `DriftManager` to monitor error signals contextually. When normal loss signals spike beyond a specific delta, the system detects "Drift" and dynamically triggers the complex adaptation strategies.

### Catastrophic Forgetting
When standard neural networks learn subsequent new tasks using gradient descent, they blindly overwrite internal weights optimized for older tasks, causing performance on those old tasks to plummet. SEAI's core architecture specifically prevents this mathematical amnesia.

### Elastic Weight Consolidation (EWC)
EWC serves as the primary continual learning regularization technique. It calculates which individual neurons are fundamentally critical to prior baseline tasks using a Fisher Information Matrix. A targeted mathematical penalty is then applied to the loss function, aggressively sheltering these critical neurons during the learning of new behaviors.

### Experience Replay Buffer
To rapidly stabilize the model upon executing an adaptation event, SEAI stores an extremely small, highly-representative cache of continuous state data. By replaying around 500 vital sample chunks alongside the new data, the model adapts violently to new trends without destabilizing old anchors, remaining extremely computationally lightweight.

---

##  System Architecture & Technologies

### Technological Stack
* **PyTorch:** The active core machine learning engine. It manages the multi-layer neural networks, calculates gradients, and processes the Fisher matrices for EWC constraints.
* **Streamlit:** Powers the enterprise-grade reactive dashboard, discarding manual terminal interfaces for sleek analytical metric monitoring in real-time.
* **Pandas & NumPy:** The mathematical foundation fueling the synthetic stream generators and continuous data structures.

### The Neural Network
Resource efficiency is prioritized by utilizing a highly-optimized **Multi-Layer Perceptron (MLP)** over massive bloated frameworks. The architecture uses dynamic inputs mapped directly via 48 active hidden nodes precisely constrained by our EWC algorithms to retain memory structurally without scaling hardware requirements.

### Supported Drift Scenarios
The framework features a custom Data Generator explicitly evaluating three environments:
1. **Sudden Drift:** Violent, instant rule permutations (e.g., zero-day cyberattacks).
2. **Gradual Drift:** Slow, prolonged rule transitions (e.g., compounding customer inflation trends).
3. **Recurring Drift:** Alternating, repeating pattern sequences (e.g., cyclic seasonal events).

### Interactive Telemetry Topologies (Metrics)
Through the analytics pipeline, SEAI captures structural telemetry:
- **Performance Trajectories:** Smoothing continuous accuracy across Baseline vs. Adaptive architectures.
- **Architectural Edge Tracking:** Graphing the explicit mathematical differential between the models at any point in time.
- **Loss Optimization Rates:** Monitoring the rapid decay efficiency of standard Cross-Entropy error arrays.
- **Processor Latency Tracking:** Proving real-time computational constraint isolation (via sparse activation logic).
- **Catastrophic Forgetting Vector:** Documenting the strict structural accuracy degradation limit bounded by the replay algorithms.

---

##  Step 1: Project Setup & Installation

Setup a contained environment utilizing `venv` to securely run the analytics pipeline locally.

### 1. Clone the Repository
```bash
git clone <repository_url_here>
cd seai_project
```

### 2. Create and Activate Virtual Environment
**Create:**
```bash
python -m venv .venv
```
**Activate (Windows PowerShell):**
```bash
.\.venv\Scripts\Activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

##  Step 2: Execution & Running the Dashboard

To launch the graphical web dashboard for immediate anomaly monitoring: 

```bash
streamlit run ui/streamlit_app.py
```
*Your browser will launch `http://localhost:8501`. Navigate the sidebar to execute simulations visually in real-time.*

---

##  Step 3: Terminal Back-End (CLI)

To bypass the web interface and purely generate native matrix calculations into the `results/` payload:

```bash
# Execute a Gradual Drift sequence (200 streaming chunks)
python main.py --scenario gradual --steps 200

# Execute a Sudden Drift sequence (500 chunks)
python main.py --scenario sudden --steps 500

# Execute a Recurring Drift scenario (300 chunks)
python main.py --scenario recurring --steps 300
```
*The metrics will naturally populate as CSV datasets within `logs/csv/`, and comparative telemetry graphs within `results/baseline_vs_seai_comparisons/`.*

---

##  Step 4: Academic Unit Testing Battery

Mathematical testing guarantees continuous validity of the EWC and Replay structures during modifications.

**Run All Validations:**
```bash
pytest tests/ -v
```

**Execute Specific Modular Components:**
```bash
# Drift Detection Layer
pytest tests/test_drift_manager.py
pytest tests/test_drift.py

# Catastrophic Forgetting Shield Core
pytest tests/test_ewc.py
pytest tests/test_forgetting_ewc.py

# Primary Neural Network
pytest tests/test_mlp.py

# Memory Retention Cache Array
pytest tests/test_replay.py

# Active Feed Simulation logic
pytest tests/test_stream.py
pytest tests/test_stream_loader.py

# Main Engine Execution Wrappers
pytest tests/test_adaptation_loop.py
pytest tests/test_trainer.py
pytest tests/test_logger.py
```

**Note:** All test commands display detailed execution logs showing:
- Test initialization details
- Input/output shapes and values
- Model parameters and gradients
- Training statistics (loss, accuracy)
- Drift detection and adaptation information
- Batch processing status

The logging is embedded in each test file, so just run the commands above to see the full details of how tests are executing!

