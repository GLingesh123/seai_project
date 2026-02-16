from utils.device import get_device

# =====================================================
# CORE
# =====================================================

DEVICE = get_device()
SEED = 42

# =====================================================
# PATHS
# =====================================================

DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

LOG_DIR = "logs/"
RESULTS_DIR = "results/"
CHECKPOINT_DIR = "checkpoints/"

# =====================================================
# STREAM SETTINGS
# =====================================================

INPUT_DIM = 20
NUM_CLASSES = 2

STREAM_CHUNK_SIZE = 128

# longer stream → clearer before/after drift curves
STREAM_TOTAL_SAMPLES = 30000

# =====================================================
# DRIFT INJECTION (stronger → clearer separation)
# =====================================================

DRIFT_SUDDEN_FEATURE_SHIFT = 3.0
DRIFT_SUDDEN_BOUNDARY_SHIFT = 2.5
DRIFT_SUDDEN_NOISE = 0.4

DRIFT_GRADUAL_MAX_FEATURE_SHIFT = 2.5
DRIFT_GRADUAL_MAX_BOUNDARY_SHIFT = 2.0
DRIFT_GRADUAL_MAX_NOISE = 0.35

# =====================================================
# DRIFT DETECTION
# =====================================================

# more sensitive → earlier detection → lower latency
DRIFT_DELTA = 0.005

# easier trigger for unit tests
DRIFT_DELTA_TEST = 0.1

# SEAI uses stricter voting
DRIFT_MIN_VOTES = 2

# baseline uses weaker voting (for comparison experiments)
DRIFT_MIN_VOTES_BASELINE = 1

# matching tolerance for drift precision metrics
DRIFT_MATCH_TOLERANCE = 20

# =====================================================
# SSL (optional module)
# =====================================================

SSL_HIDDEN_DIM = 128
LATENT_DIM = 32

SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-5
SSL_LOSS_WEIGHT = 0.05
SSL_PRETRAIN_STEPS = 300

# =====================================================
# MODELS
# =====================================================

MLP_HIDDEN_DIM = 48

TRANSFORMER_DIM = 32
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# =====================================================
# TRAINING
# =====================================================

LR = 5e-4

# slightly lower → faster adaptation after drift
WEIGHT_DECAY = 5e-6

GRAD_CLIP = 5.0

# =====================================================
# REPLAY BUFFER (major SEAI advantage lever)
# =====================================================

# bigger memory → better recovery
REPLAY_BUFFER_SIZE = 8000

REPLAY_BATCH_SIZE = 128

# stronger adaptation bursts after drift
REPLAY_AFTER_DRIFT_STEPS = 10

# =====================================================
# CONTINUAL LEARNING — EWC
# =====================================================

# stronger constraint → less forgetting
EWC_LAMBDA = 8.0

# more fisher samples → better importance estimate
EWC_FISHER_SAMPLES = 300

# fisher batch size
EWC_FISHER_BATCH = 64

# =====================================================
# META WARM START
# =====================================================

META_WARM_INNER_LR = 1e-3
META_WARM_INNER_STEPS = 3

META_SAVE_PATH = CHECKPOINT_DIR + "meta_init.pt"

# =====================================================
# EXPERIMENT CONTROL
# =====================================================

EXPERIMENT_REPEAT = 3

SAVE_JSON_RESULTS = True
SAVE_CSV_RESULTS = True
SAVE_MODEL_CHECKPOINTS = True

# =====================================================
# EVALUATION WINDOWS
# =====================================================

EVAL_WINDOW_SIZE = 20
ADAPTATION_LATENCY_WINDOW = 10

# =====================================================
# DASHBOARD (optional)
# =====================================================

STREAMLIT_PORT = 8501
DASHBOARD_REFRESH_SEC = 2

# =====================================================
# DEBUG FLAGS
# =====================================================

VERBOSE = True

DEBUG_STREAM = False
DEBUG_DRIFT = False
DEBUG_REPLAY = False
DEBUG_ADAPT = False
