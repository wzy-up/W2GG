import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "W2GG"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""

# Pseudo-glance preprocessing switches used by the ablation studies.
_C.DATASETS.PSEUDO_GLANCE = CN()
# Which pseudo-keyframe selector family to use when SELECTED_FRAMES_KEY is set.
# Canonical values: "rba" or "original_aks".
# Aliases such as "rab" and "modified_aks" are accepted by paths_catalog.py.
_C.DATASETS.PSEUDO_GLANCE.SELECTOR = "rba"
# "single": keep the highest-confidence pseudo glance only.
# "multi": keep multiple pseudo glances.
_C.DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE = "multi"
# 0 means use every pseudo glance stored in selected_frames.json.
_C.DATASETS.PSEUDO_GLANCE.MAX_KEYFRAMES = 0
# Select an independently generated selector output configured in paths_catalog.py,
# e.g. "k1", "k4", or "k16". Empty means use the dataset default path.
_C.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY = ""
# Optional explicit path override. It has higher priority than SELECTED_FRAMES_KEY.
_C.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_FILE = ""
# "none", "dedup", or "dedup_filter".
_C.DATASETS.PSEUDO_GLANCE.REFINEMENT = "dedup_filter"
# Filtering threshold = FILTER_THRESHOLD_SCALE / number of raw keyframes.
_C.DATASETS.PSEUDO_GLANCE.FILTER_THRESHOLD_SCALE = 1.0
# "confidence" keeps selector scores; "uniform" assigns equal weights.
_C.DATASETS.PSEUDO_GLANCE.WEIGHT_MODE = "confidence"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.W2GG = CN()
_C.MODEL.W2GG.NUM_CLIPS = 128
_C.MODEL.W2GG.JOINT_SPACE_SIZE = 256

_C.MODEL.W2GG.FEATPOOL = CN()
_C.MODEL.W2GG.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.W2GG.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.W2GG.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.W2GG.FEAT2D = CN()
_C.MODEL.W2GG.FEAT2D.NAME = "pool"
_C.MODEL.W2GG.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.W2GG.TEXT_ENCODER = CN()
_C.MODEL.W2GG.TEXT_ENCODER.NAME = 'BERT'

_C.MODEL.W2GG.PREDICTOR = CN() 
_C.MODEL.W2GG.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.W2GG.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.W2GG.PREDICTOR.NUM_STACK_LAYERS = 8


_C.MODEL.W2GG.LOSS = CN()
_C.MODEL.W2GG.LOSS.MIN_IOU = 0.3
_C.MODEL.W2GG.LOSS.MAX_IOU = 0.7
_C.MODEL.W2GG.LOSS.BCE_WEIGHT = 1
_C.MODEL.W2GG.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 1
_C.MODEL.W2GG.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.W2GG.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.W2GG.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.W2GG.LOSS.CONTRASTIVE_WEIGHT = 0.05
_C.MODEL.W2GG.LOSS.TAU_VIDEO = 0.2
_C.MODEL.W2GG.LOSS.TAU_SENT = 0.2
_C.MODEL.W2GG.LOSS.MARGIN = 0.2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0
# for gaussian weight generation
_C.SOLVER.SIGMA = 0.3 
_C.SOLVER.WINDOW = 4 
_C.SOLVER.TOPK = 10 
_C.SOLVER.THRESH = 0.9
_C.SOLVER.USE_DGA = False
#区间筛选counter+1还是-1
_C.SOLVER.CENTER_POS_THRESH = 0.90
_C.SOLVER.CENTER_NEG_THRESH = 0.60

# Gaussian-prior ablation switches.
_C.SOLVER.DGA = CN()
# Modes:
#   "auto"           : backward-compatible mapping from SOLVER.USE_DGA
#   "single_static"  : one fixed Gaussian centered at the top pseudo glance
#   "static_multi"   : fixed multi-Gaussian mixture over refined pseudo glances
#   "single_anchor"  : dynamic DGA driven by one highest-confidence anchor
#   "multi_mean"     : dynamic DGA driven by uniformly averaged anchors
#   "multi_weighted" : dynamic DGA driven by confidence-weighted anchors
_C.SOLVER.DGA.MODE = "auto"
_C.SOLVER.DGA.CONTINUOUS_NUM = 2
_C.SOLVER.DGA.EMA_ALPHA = 0.7
_C.SOLVER.DGA.FALLBACK_TO_TOP_ANCHOR = True


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ---------------------------------------------------------------------------- #
# better gmask2d generation
# ---------------------------------------------------------------------------- #
_C.DATASETS.PSEUDO_GMASK = CN()
# Modes: "single", "union", "confidence", "length", or "qpm".
_C.DATASETS.PSEUDO_GMASK.MODE = "qpm"
_C.DATASETS.PSEUDO_GMASK.KEEP_RATIO = 0.42
_C.DATASETS.PSEUDO_GMASK.MIN_KEEP = 24
_C.DATASETS.PSEUDO_GMASK.MARGIN = 3
