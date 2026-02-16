from utils.device import get_device

# ================= CORE =================

DEVICE = get_device()
SEED = 42

# ================= PATHS =================

DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

LOG_DIR = "logs/"
RESULTS_DIR = "results/"
CHECKPOINT_DIR = "checkpoints/"

# ================= STREAM =================

INPUT_DIM = 20
NUM_CLASSES = 2

STREAM_CHUNK_SIZE = 128
STREAM_TOTAL_SAMPLES = 20000

# ================= DRIFT INJECTION =================

DRIFT_SUDDEN_FEATURE_SHIFT = 2.0
DRIFT_SUDDEN_BOUNDARY_SHIFT = 1.5
DRIFT_SUDDEN_NOISE = 0.25

DRIFT_GRADUAL_MAX_FEATURE_SHIFT = 2.0
DRIFT_GRADUAL_MAX_BOUNDARY_SHIFT = 1.5
DRIFT_GRADUAL_MAX_NOISE = 0.3

# ================= DRIFT DETECTION =================

DRIFT_DELTA = 0.01
DRIFT_DELTA_TEST = 0.1

DRIFT_MIN_VOTES = 2
DRIFT_MIN_VOTES_BASELINE = 1
DRIFT_MATCH_TOLERANCE = 20

# ================= SSL =================

SSL_HIDDEN_DIM = 128
LATENT_DIM = 32

SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-5
SSL_LOSS_WEIGHT = 0.05
SSL_PRETRAIN_STEPS = 300

# ================= MODELS =================

MLP_HIDDEN_DIM = 64

TRANSFORMER_DIM = 32
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

# ================= TRAINING =================

LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

# ================= REPLAY =================

REPLAY_BUFFER_SIZE = 5000
REPLAY_BATCH_SIZE = 128
REPLAY_AFTER_DRIFT_STEPS = 6

# ================= EWC =================

EWC_LAMBDA = 5.0
EWC_FISHER_SAMPLES = 200

# ================= META =================

META_WARM_INNER_LR = 1e-3
META_WARM_INNER_STEPS = 3

META_SAVE_PATH = CHECKPOINT_DIR + "meta_init.pt"

# ================= EXPERIMENT =================

EXPERIMENT_REPEAT = 3

# ================= DEBUG =================

VERBOSE = True
# =====================================================
# Debug / Verbosity Controls
# =====================================================

VERBOSE = True

DEBUG_STREAM = False
DEBUG_DRIFT = False
DEBUG_REPLAY = False
DEBUG_ADAPT = False
# =====================================================
# Continual Learning — EWC
# =====================================================

EWC_LAMBDA = 5.0

# number of batches used to estimate Fisher
EWC_FISHER_SAMPLES = 200

# batch size per Fisher sample draw
EWC_FISHER_BATCH = 64
