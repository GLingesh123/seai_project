"""
SEAI Project Global Configuration
All tunable parameters must live here.
Do NOT scatter constants across modules.
"""

from utils.device import get_device


# =====================================================
# Core Runtime
# =====================================================

DEVICE = get_device()
SEED = 42


# =====================================================
# Paths
# =====================================================

DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

LOG_DIR = "logs/"
RESULTS_DIR = "results/"
CHECKPOINT_DIR = "checkpoints/"


# =====================================================
# Synthetic Stream Settings
# =====================================================

INPUT_DIM = 20
NUM_CLASSES = 2

STREAM_CHUNK_SIZE = 128
STREAM_TOTAL_SAMPLES = 20000

STREAM_BASE_MEAN = 0.0
STREAM_BASE_STD = 1.0


# =====================================================
# Drift Settings (used by drift injector / scenarios)
# =====================================================

DRIFT_SUDDEN_FEATURE_SHIFT = 2.0
DRIFT_SUDDEN_BOUNDARY_SHIFT = 1.5
DRIFT_SUDDEN_NOISE = 0.25

DRIFT_GRADUAL_MAX_FEATURE_SHIFT = 2.0
DRIFT_GRADUAL_MAX_BOUNDARY_SHIFT = 1.5
DRIFT_GRADUAL_MAX_NOISE = 0.3


# =====================================================
# Self-Supervised Learning (Autoencoder)
# =====================================================

SSL_HIDDEN_DIM = 64
LATENT_DIM = 16

SSL_LR = 1e-3
SSL_EPOCHS = 5


# =====================================================
# Baseline Model (MLP)
# =====================================================

MLP_HIDDEN_DIM = 64


# =====================================================
# Transformer Model
# =====================================================

TRANSFORMER_DIM = 32
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1


# =====================================================
# Training
# =====================================================

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5

MAX_STEPS = STREAM_TOTAL_SAMPLES // STREAM_CHUNK_SIZE


# =====================================================
# Replay Buffer (Continual Learning)
# =====================================================

REPLAY_BUFFER_SIZE = 500
REPLAY_BATCH_SIZE = 128
REPLAY_AFTER_DRIFT_STEPS = 2


# =====================================================
# Continual Learning — EWC
# =====================================================

EWC_LAMBDA = 0.4
EWC_FISHER_SAMPLES = 256


# =====================================================
# Drift Detector (River ADWIN)
# =====================================================

DRIFT_DELTA = 0.002


# =====================================================
# Meta Initialization
# =====================================================

META_PRETRAIN_ROUNDS = 3
META_SAVE_PATH = CHECKPOINT_DIR + "meta_init.pt"


# =====================================================
# Experiment Settings
# =====================================================

EXPERIMENT_REPEAT = 3
SAVE_JSON_RESULTS = True
SAVE_MODEL_CHECKPOINTS = True


# =====================================================
# Dashboard
# =====================================================

STREAMLIT_PORT = 8501
DASHBOARD_REFRESH_SEC = 2


# =====================================================
# Debug
# =====================================================

VERBOSE = True
