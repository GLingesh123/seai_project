"""
UNIFIED TEST FILE: SEAI vs Baseline
=====================================
Complete end-to-end test that runs both baseline and SEAI,
generates comprehensive analysis, and organizes results by metric.

One command to run full experiment and generate all results.

Usage:
    python main.py --scenario gradual
    python main.py --scenario sudden --steps 500
    python main.py --scenario recurring
    python main.py --scenario none
"""

import os
import sys
import argparse
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from data.stream_loader import StreamLoader
from models.mlp import BaselineMLP
from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop
from drift.drift_manager import DriftManager
from continual_learning.replay_buffer import ReplayBuffer
from continual_learning.ewc import EWC
from continual_learning.ewc import EWC
from continual_learning.ssl_contrastive import ContrastiveSSL
from utils.logger import ExperimentLogger

from config import RESULTS_DIR, DRIFT_MIN_VOTES, EWC_LAMBDA, LOG_DIR


# =====================================================
# RESULT ORGANIZATION
# =====================================================

class ResultOrganizer:
    """Organizes visual evaluation arrays exclusively into baseline_vs_seai_comparisons"""
    
    def __init__(self, base_dir=RESULTS_DIR):
        self.base_dir = base_dir
        self.metric_dirs = {
            'comparisons': f"{base_dir}/baseline_vs_seai_comparisons"
        }
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create all target directories"""
        for dir_path in self.metric_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def save_metric_file(self, folder: str, filename: str, df: pd.DataFrame):
        """Save a metric file (e.g., accuracy.csv)"""
        if folder not in self.metric_dirs:
            raise ValueError(f"Unknown folder: {folder}")
        
        path = f"{self.metric_dirs[folder]}/{filename}"
        df.to_csv(path, index=False)
        return path
    
    def save_analysis_file(self, folder: str, filename: str, content: str):
        """Save analysis text file"""
        path = f"{self.metric_dirs[folder]}/{filename}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def save_plot(self, folder: str, filename: str, fig=None):
        """Save matplotlib figure"""
        path = f"{self.metric_dirs[folder]}/{filename}"
        if fig:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return path


# =====================================================
# BASELINE RUNNER
# =====================================================

def run_baseline(scenario: dict, steps: int):
    """
    Run static baseline (no adaptation)
    
    Train only before drift, then evaluate post-drift
    """
    print("\n" + "="*80)
    print("🏃 RUNNING BASELINE")
    print("="*80)
    
    rows = []
    drift_step = scenario.get("start", scenario.get("steps", [30])[0] if isinstance(scenario.get("steps"), list) else 30)
    
    # ---------- Train stream (before drift) ----------
    train_stream = StreamLoader(scenario=scenario, seed=42, total_samples=steps * 100 + 5000)
    
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    import time
    
    # Train only before drift
    for step in range(drift_step):
        batch = train_stream.next_batch()
        if batch is None:
            break
        
        X, y, _ = batch
        
        start_t = time.time()
        stats = trainer.train_batch(X, y)
        t_ms = (time.time() - start_t) * 1000
        mem_kb = model.num_parameters() * 4 / 1024
        
        rows.append({
            "step": step,
            "loss": stats["loss"],
            "accuracy": stats["accuracy"],
            "drift": False,
            "time_ms": t_ms,
            "memory_kb": mem_kb
        })
        
        if (step + 1) % 50 == 0:
            print(f"  Train step {step + 1}: acc={stats['accuracy']:.4f}, loss={stats['loss']:.4f}")
    
    # ---------- Eval stream (after drift) ----------
    eval_stream = StreamLoader(scenario=scenario, seed=9999, total_samples=steps * 100 + 5000)
    
    # Skip pre-drift region
    for _ in range(drift_step):
        eval_stream.next_batch()
    
    # Evaluate post-drift
    for step in range(drift_step, steps):
        batch = eval_stream.next_batch()
        if batch is None:
            break
        
        X, y, _ = batch
        
        start_t = time.time()
        acc = trainer.eval_batch(X, y)
        t_ms = (time.time() - start_t) * 1000
        mem_kb = model.num_parameters() * 4 / 1024
        
        with torch.no_grad():
            from config import DEVICE
            tX = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
            ty = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
            loss_val = F.cross_entropy(model(tX), ty).item()
            
        if scenario.get("type") in ["sudden", "recurring"]:
            is_new_drift = (step in scenario.get("steps", []))
        elif scenario.get("type") == "gradual":
            is_new_drift = (step == scenario.get("start", -1))
        else:
            is_new_drift = False
        
        rows.append({
            "step": step,
            "loss": loss_val,
            "accuracy": acc,
            "drift": is_new_drift,
            "time_ms": t_ms,
            "memory_kb": mem_kb
        })
        
        if (step + 1) % 50 == 0:
            print(f"  Eval step {step + 1}: acc={acc:.4f}")
    
    df = pd.DataFrame(rows)
    
    baseline_path = f"{LOG_DIR}/csv/baseline_test_run.csv"
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    df.to_csv(baseline_path, index=False)
    
    print(f"\n✅ Baseline complete: {len(df)} steps")
    print(f"  Final Accuracy: {df['accuracy'].iloc[-1]:.4f}")
    print(f"  Avg Accuracy: {df['accuracy'].mean():.4f}")
    
    return df


# =====================================================
# SEAI RUNNER
# =====================================================

def run_seai(scenario: dict, steps: int):
    """
    Run SEAI (adaptive learning with drift detection)
    """
    print("\n" + "="*80)
    print("🤖 RUNNING SEAI (Adaptive)")
    print("="*80)
    
    stream = StreamLoader(scenario=scenario, total_samples=steps * 100 + 5000)
    
    model = BaselineMLP()
    # Initialize the Self-Supervised Contrastive Loss module (Eq. 5)
    ssl_module = ContrastiveSSL(model=model, temperature=0.1, noise_std=0.05)
    
    trainer = StreamTrainer(model, ssl_module=ssl_module, ssl_weight=0.1)
    
    detector = DriftManager(min_votes=DRIFT_MIN_VOTES)
    replay = ReplayBuffer()
    continual = EWC(trainer.model, lambda_ewc=EWC_LAMBDA)
    
    logger = ExperimentLogger("seai_test_run")
    
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        continual_module=continual,
        logger=logger
    )
    
    loop.run(max_steps=steps)
    
    df = pd.read_csv(f"{LOG_DIR}/csv/{logger.run_id}.csv")
    print(f"\n✅ SEAI complete: {len(df)} steps")
    print(f"  Final Accuracy: {df['accuracy'].iloc[-1]:.4f}")
    print(f"  Avg Accuracy: {df['accuracy'].mean():.4f}")
    print(f"  Drift Events: {len(df[df['drift'] == True])}")
    
    return df


# =====================================================
# ANALYSIS & COMPARISON
# =====================================================

def generate_accuracy_comparison(baseline_df: pd.DataFrame, seai_df: pd.DataFrame) -> pd.DataFrame:
    """Compare accuracy between baseline and SEAI"""
    comparison = pd.DataFrame({
        'step': baseline_df['step'],
        'baseline_accuracy': baseline_df['accuracy'],
        'seai_accuracy': seai_df['accuracy'],
        'improvement': seai_df['accuracy'] - baseline_df['accuracy'],
        'improvement_pct': ((seai_df['accuracy'] - baseline_df['accuracy']) / 
                           (baseline_df['accuracy'].abs() + 1e-6) * 100)
    })
    return comparison


def generate_loss_comparison(baseline_df: pd.DataFrame, seai_df: pd.DataFrame) -> pd.DataFrame:
    """Compare loss between baseline and SEAI"""
    comparison = pd.DataFrame({
        'step': baseline_df['step'],
        'baseline_loss': baseline_df['loss'],
        'seai_loss': seai_df['loss'],
        'loss_reduction': baseline_df['loss'] - seai_df['loss'],
        'loss_reduction_pct': ((baseline_df['loss'] - seai_df['loss']) / 
                               (baseline_df['loss'].abs() + 1e-6) * 100)
    })
    return comparison


def generate_improvement_report(baseline_df: pd.DataFrame, seai_df: pd.DataFrame, scenario: dict) -> str:
    """Generate detailed improvement analysis for the UI text parser"""
    
    a_base = baseline_df['accuracy'].mean()
    a_seai = seai_df['accuracy'].mean()
    
    # 4 Core Targeted Custom Metrics
    t_base = baseline_df['time_ms'].mean()
    t_seai = seai_df['time_ms'].mean()
    
    m_base = baseline_df['memory_kb'].max()
    m_seai = seai_df['memory_kb'].max()
    
    f_base = baseline_df['accuracy'].max() - baseline_df['accuracy'].iloc[-1]
    f_seai = seai_df['accuracy'].max() - seai_df['accuracy'].iloc[-1]
    
    report = f"""
================================================================================
SEAI vs BASELINE - COMPREHENSIVE ANALYSIS
================================================================================
A_BASE: {a_base:.4f}
A_SEAI: {a_seai:.4f}
T_BASE: {t_base:.4f}
T_SEAI: {t_seai:.4f}
M_BASE: {m_base:.4f}
M_SEAI: {m_seai:.4f}
F_BASE: {f_base:.4f}
F_SEAI: {f_seai:.4f}
================================================================================
"""
    return report

def generate_concise_summary(baseline_df, seai_df, pred_steps):
    a_base = baseline_df['accuracy'].mean()
    a_seai = seai_df['accuracy'].mean()
    
    summary = f"""
================================================================================
SEAI vs BASELINE - SUMMARY REPORT
================================================================================
[BASELINE] Final Acc: {baseline_df['accuracy'].iloc[-1]:.2%}, Avg Acc: {a_base:.2%}
[  SEAI  ] Final Acc: {seai_df['accuracy'].iloc[-1]:.2%}, Avg Acc: {a_seai:.2%}

★ ACCURACY BOOST: SEAI improved average accuracy by +{(a_seai - a_base)*100:.2f}%
★ DRIFT DETECTED: {len(pred_steps)} event(s)
================================================================================
"""
    return summary


# =====================================================
# VISUALIZATION
# =====================================================

def plot_drift_window(plt_mod, scenario: dict):
    if not scenario or scenario.get("type") == "none":
        return
    if scenario.get("type") == "gradual":
        start = scenario.get("start", 30)
        end = scenario.get("end", 80)
        plt_mod.axvspan(start, end, color='red', alpha=0.15, label='Drift Window')
    else:
        steps = scenario.get("steps", [30])
        for step in steps:
            plt_mod.axvline(x=step, color='red', linestyle='--', alpha=0.5, label='True Drift' if step == steps[0] else "")

def plot_comparison_metrics(baseline_df: pd.DataFrame, seai_df: pd.DataFrame, scenario: dict, organizer: ResultOrganizer):
    """Plot enterprise-grade, highly stylized accuracy and loss comparison graphs."""
    plt.style.use('dark_background')
    
    # Apply rolling smoothing for professional aesthetics
    smooth_w = max(1, len(baseline_df) // 20)
    b_acc_s = baseline_df['accuracy'].rolling(window=smooth_w, min_periods=1).mean()
    s_acc_s = seai_df['accuracy'].rolling(window=smooth_w, min_periods=1).mean()
    b_loss_s = baseline_df['loss'].rolling(window=smooth_w, min_periods=1).mean()
    s_loss_s = seai_df['loss'].rolling(window=smooth_w, min_periods=1).mean()
    
    def add_drift_markers(ax):
        if scenario and scenario.get("type") in ["sudden", "recurring"]:
            for d_step in scenario.get("steps", []):
                if d_step < len(baseline_df):
                    ax.axvline(x=d_step, color='#f1c40f', linestyle=':', alpha=0.8, linewidth=2)
                    ax.text(d_step + int(len(baseline_df)*0.02), ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 'DRIFT INITIATED', color='#f1c40f', fontsize=9, fontweight='bold', rotation=90)
        elif scenario and scenario.get("type") == "gradual":
            start_d = scenario.get("start", 0)
            end_d = scenario.get("end", 0)
            if start_d < len(baseline_df):
                ax.axvspan(start_d, min(end_d, len(baseline_df)), color='#f1c40f', alpha=0.1)
                ax.text(start_d + int(len(baseline_df)*0.02), ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 'GRADUAL DRIFT', color='#f1c40f', fontsize=9, fontweight='bold', rotation=90)

    def style_axis(ax, title, ylabel):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.7)
        ax.set_xlabel('Execution Trace (Steps)', fontsize=11, color='#aaaaaa', fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, color='#aaaaaa', fontweight='bold')
        ax.set_title(title, fontsize=15, pad=20, color='white', fontweight='900')
        ax.tick_params(colors='#aaaaaa')
        ax.legend(frameon=True, facecolor='#111111', edgecolor='#333333', fontsize=10)

    # 1. Smoothed Accuracy Trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(baseline_df['step'], b_acc_s, label='Baseline Matrix (Rigid)', color='#ff4b4b', linewidth=2.5)
    ax1.plot(seai_df['step'], s_acc_s, label='SEAI Engine (Adaptive)', color='#00f2fe', linewidth=2.5)
    ax1.fill_between(baseline_df['step'], 0, b_acc_s, color='#ff4b4b', alpha=0.05)
    ax1.fill_between(seai_df['step'], 0, s_acc_s, color='#00f2fe', alpha=0.1)
    style_axis(ax1, 'Performance Trajectory over Continuous Time', 'Smoothed Accuracy Ratio')
    add_drift_markers(ax1)
    organizer.save_plot('comparisons', 'accuracy_comparison.jpg', fig1); plt.close(fig1)

    # 2. Smoothed Loss Dynamics
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(baseline_df['step'], b_loss_s, label='Baseline Error Function', color='#ff4b4b', linewidth=2.5)
    ax2.plot(seai_df['step'], s_loss_s, label='SEAI Error Function', color='#00f2fe', linewidth=2.5)
    ax2.fill_between(baseline_df['step'], 0, b_loss_s, color='#ff4b4b', alpha=0.05)
    ax2.fill_between(seai_df['step'], 0, s_loss_s, color='#00f2fe', alpha=0.1)
    style_axis(ax2, 'Mathematical Loss Decay Optimization', 'Smoothed Cross-Entropy Loss')
    add_drift_markers(ax2)
    organizer.save_plot('comparisons', 'loss_comparison.jpg', fig2); plt.close(fig2)

    # 3. Structural Accuracy Advantage Tracker
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    adv = s_acc_s - b_acc_s
    ax3.plot(baseline_df['step'], adv, label='SEAI Cumulative Advantage', color='#2ecc71', linewidth=2)
    ax3.fill_between(baseline_df['step'], 0, adv, where=(adv >= 0), color='#2ecc71', alpha=0.2, label='SEAI Positive Delta')
    ax3.fill_between(baseline_df['step'], 0, adv, where=(adv < 0), color='#ff4b4b', alpha=0.2, label='Baseline Positive Delta')
    style_axis(ax3, 'Net Architectural Edge (Accuracy Delta)', 'SEAI % Advantage')
    add_drift_markers(ax3)
    organizer.save_plot('comparisons', 'accuracy_advantage.jpg', fig3); plt.close(fig3)

    # 4. Latency Resource Spike Tracker
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    b_time = baseline_df['time_ms'].rolling(window=smooth_w, min_periods=1).mean() if smooth_w > 1 else baseline_df['time_ms']
    s_time = seai_df['time_ms'].rolling(window=smooth_w, min_periods=1).mean() if smooth_w > 1 else seai_df['time_ms']
    ax4.plot(baseline_df['step'], b_time, label='Baseline Compute Overhead', color='#ff4b4b', linewidth=2)
    ax4.plot(seai_df['step'], s_time, label='SEAI Compute Overhead', color='#00f2fe', linewidth=2)
    ax4.fill_between(seai_df['step'], 0, s_time, color='#00f2fe', alpha=0.1)
    style_axis(ax4, 'On-Demand Processor Activation (Latency)', 'Compute Cycles (ms)')
    add_drift_markers(ax4)
    organizer.save_plot('comparisons', 'latency_comparison.jpg', fig4); plt.close(fig4)

    # 5. Dynamic Catastrophic Forgetting Tracker
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    b_forget = (baseline_df['accuracy'].cummax() - baseline_df['accuracy']).rolling(window=smooth_w, min_periods=1).mean()
    s_forget = (seai_df['accuracy'].cummax() - seai_df['accuracy']).rolling(window=smooth_w, min_periods=1).mean()
    
    ax5.plot(baseline_df['step'], b_forget, label='Baseline Forgetting Magnitude', color='#ff4b4b', linewidth=2.5)
    ax5.plot(seai_df['step'], s_forget, label='SEAI Forgetting Magnitude', color='#00f2fe', linewidth=2.5)
    ax5.fill_between(baseline_df['step'], 0, b_forget, color='#ff4b4b', alpha=0.1)
    ax5.fill_between(seai_df['step'], 0, s_forget, color='#00f2fe', alpha=0.15)
    
    style_axis(ax5, 'Catastrophic Forgetting Vector (Closer to 0 = Complete Retention)', 'Accuracy Degradation Gap')
    add_drift_markers(ax5)
    organizer.save_plot('comparisons', 'forgetting_comparison.jpg', fig5); plt.close(fig5)
    
    # 6. Drift specific accuracy (Zoomed in plot)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    drift_center = len(baseline_df) // 2
    if scenario and scenario.get("steps"):
        drift_center = scenario.get("steps")[0]
    elif scenario and scenario.get("start"):
        drift_center = scenario.get("start")
    
    start_idx = max(0, drift_center - 20)
    end_idx = min(len(baseline_df), drift_center + 60)
    
    ax6.plot(baseline_df['step'][start_idx:end_idx], b_acc_s[start_idx:end_idx], label='Baseline Matrix', color='#ff4b4b', linewidth=2.5)
    ax6.plot(seai_df['step'][start_idx:end_idx], s_acc_s[start_idx:end_idx], label='SEAI Engine', color='#00f2fe', linewidth=2.5)
    style_axis(ax6, 'Post-Drift Specific Accuracy Recovery', 'Accuracy Ratio')
    add_drift_markers(ax6)
    organizer.save_plot('comparisons', 'drift_accuracy.jpg', fig6); plt.close(fig6)

    # 7. Forgetting Rate (Derivative of Forgetting)
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    b_forget_rate = b_forget.diff().fillna(0).rolling(window=smooth_w, min_periods=1).mean()
    s_forget_rate = s_forget.diff().fillna(0).rolling(window=smooth_w, min_periods=1).mean()
    ax7.plot(baseline_df['step'], b_forget_rate, label='Baseline Forgetting Rate', color='#ff4b4b', linewidth=2)
    ax7.plot(seai_df['step'], s_forget_rate, label='SEAI Forgetting Rate', color='#00f2fe', linewidth=2)
    ax7.axhline(0, color='#aaaaaa', linestyle='--', alpha=0.5)
    style_axis(ax7, 'Dynamic Forgetting Rate Optimization', 'Rate of Memory Loss')
    add_drift_markers(ax7)
    organizer.save_plot('comparisons', 'forgetting_rate.jpg', fig7); plt.close(fig7)

    # 8. Adaptation Time
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    ax8.bar(baseline_df['step'], baseline_df['time_ms'], label='Baseline Compute', color='#ff4b4b', alpha=0.6)
    ax8.plot(seai_df['step'], seai_df['time_ms'], label='SEAI Compute', color='#00f2fe', linewidth=2)
    style_axis(ax8, 'Adaptation Recovery Timing', 'Compute Cycles (ms)')
    add_drift_markers(ax8)
    organizer.save_plot('comparisons', 'adaptation_time.jpg', fig8); plt.close(fig8)

    # 9. Memory Resource Usage
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    ax9.plot(baseline_df['step'], baseline_df['memory_kb'], label='Baseline Memory', color='#ff4b4b', linewidth=2)
    ax9.plot(seai_df['step'], seai_df['memory_kb'], label='SEAI Memory Buffer', color='#00f2fe', linewidth=2)
    style_axis(ax9, 'RAM & Virtual Memory Sizing', 'Memory Usage (KB)')
    add_drift_markers(ax9)
    organizer.save_plot('comparisons', 'memory_usage.jpg', fig9); plt.close(fig9)
    
    plt.style.use('default')


# =====================================================
# MAIN EXECUTION
# =====================================================

def get_scenario(name: str):
    """Get scenario configuration"""
    scenarios = {
        "none": {"type": "none"},
        "sudden": {"type": "sudden", "steps": [30]},
        "gradual": {"type": "gradual", "start": 30, "end": 80},
        "recurring": {"type": "recurring", "steps": [25, 70, 110]}
    }
    
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}")
    
    return scenarios[name]


def main(args):
    """Main test execution"""
    
    print("\n" + "="*80)
    print("SEAI UNIFIED TEST - Complete Baseline vs SEAI Comparison")
    print("="*80)
    print(f"Scenario: {args.scenario.upper()}")
    print(f"Steps: {args.steps}")
    print("="*80)
    
    # Setup
    scenario = get_scenario(args.scenario)
    organizer = ResultOrganizer()
    
    # Run both
    baseline_df = run_baseline(scenario, args.steps)
    seai_df = run_seai(scenario, args.steps)
    
    # Generate overarching comparison pipelines
    
    acc_comp = generate_accuracy_comparison(baseline_df, seai_df)
    loss_comp = generate_loss_comparison(baseline_df, seai_df)
    
    plot_comparison_metrics(baseline_df, seai_df, scenario, organizer)
    report = generate_improvement_report(baseline_df, seai_df, scenario)
    
    pred_steps = seai_df[seai_df['drift'] == True]['step'].tolist() if 'drift' in seai_df.columns else []
    concise_summary = generate_concise_summary(baseline_df, seai_df, pred_steps)
    
    # Print exclusively the compact summary visually
    print("\n" + concise_summary)
    
    # Broadcast the comprehensive mathematics to Streamlit via a masked sequence
    print("___HIDDEN_REPORT_START___")
    print(report)
    print("___HIDDEN_REPORT_END___")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SEAI Unified Test')
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='gradual',
        choices=['none', 'sudden', 'gradual', 'recurring'],
        help='Drift scenario'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Number of steps'
    )
    
    args = parser.parse_args()
    main(args)
