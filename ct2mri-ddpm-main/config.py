"""
Central configuration for the CT -> MRI diffusion project.

Every script imports its settings from here. Edit the constants below rather
than passing command-line arguments. The only runtime override is the
CT2MRI_DATA_ROOT environment variable, so the same code can point at either
the server path or a locally mounted copy (e.g. via sshfs) without edits.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Root folder holding the cross-validation datasets on the server.
DATA_ROOT = os.environ.get("CT2MRI_DATA_ROOT", "/media/aisec-102/DATA3/rachel/data/CV")

# Sub-folder naming follows the prior work: "{equalization}_{mask}".
#   equalization: "eq", "no_eq", "avg_eq"
#   mask:         "paired" (no mask), "seg", "bin_edge"
EQUALIZATION = "avg_eq"
MASK_TYPE = "paired"

# Whether to feed the mask (arr_2) to the network as an extra condition channel.
# Only valid when MASK_TYPE is not "paired".
USE_MASK = MASK_TYPE != "paired"

# File prefix inside the dataset folder; the fold number and ".npz" are appended.
FILE_PREFIX = "norm_mask_neg1pos1_fold" if MASK_TYPE != "paired" else "norm_neg1pos1_fold"

DATASET_NAME = f"{EQUALIZATION}_{MASK_TYPE}"
DATA_PATH_PREFIX = os.path.join(DATA_ROOT, DATASET_NAME, FILE_PREFIX)

# Five-fold cross-validation: train on the other four, evaluate on this one.
NUM_FOLDS = 5
TEST_FOLD = 5
TRAIN_FOLDS = [f for f in range(1, NUM_FOLDS + 1) if f != TEST_FOLD]

IMAGE_SIZE = 256

# Augmentation (training only). Applied identically to CT, MRI and mask.
AUGMENT = True
AUG_HFLIP_P = 0.5
AUG_VFLIP_P = 0.0
AUG_ROT90_P = 0.0

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Physical GPU 0 on the server is enumerated by PyTorch as cuda:1.
DEVICE = "cuda:1"
NUM_WORKERS = 4
USE_AMP = True            # bfloat16 autocast on CUDA
USE_COMPILE = False       # torch.compile the UNet
SEED = 42

# ---------------------------------------------------------------------------
# Diffusion / noise scheduler
# ---------------------------------------------------------------------------

NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE = "linear"  # "linear" or "cosine"
BETA_START = 1e-4
BETA_END = 2e-2
PREDICTION_TYPE = "eps"   # "eps" (predict noise) or "v" (v-prediction)

# Sampling used for validation and testing.
SAMPLER = "ddim"          # "ddim" or "ddpm"
SAMPLING_STEPS = 50       # ignored for "ddpm", which always uses every step
DDIM_ETA = 0.0            # 0 = deterministic DDIM

# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

IMG_CHANNELS = 1                                  # MRI channels being generated
COND_CHANNELS = 1 + (1 if USE_MASK else 0)        # CT (+ mask)
BASE_CHANNELS = 64
CHANNEL_MULT = (1, 1, 2, 2, 4, 4)                 # 256 -> 8 over six levels
NUM_RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = (16, 8)
NUM_HEADS = 4
DROPOUT = 0.1
GROUP_NORM_GROUPS = 32

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

LOSS_TYPE = "mse"         # "mse", "l1" or "huber"
MIN_SNR_GAMMA = None      # e.g. 5.0 to enable Min-SNR loss weighting

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
EPOCHS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 1000
LR_SCHEDULE = "constant"  # "constant" or "cosine" (after warmup)
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995

RESUME = True             # continue from last.pt in the run folder if present
CKPT_EVERY = 5            # epochs between overwriting last.pt
VAL_EVERY = 10            # epochs between sampled validation runs
VAL_NUM_SAMPLES = 16      # test-fold images sampled during validation
NUM_PLOT_SAMPLES = 4      # images shown in the validation figure

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE = 16
COMPUTE_FID = True
SAVE_PREDICTIONS = True   # store generated MRIs as a .npz for later use
NUM_TEST_FIGURES = 20     # CT | rMRI | sMRI figures saved during testing
# Checkpoint in the run's checkpoint folder evaluated by test.py. "best.pt" is
# selected on a subset of the test fold, so "final.pt" gives the unbiased number.
CHECKPOINT_NAME = "final.pt"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "runs"

# Optional suffix for ablations (e.g. "v_cosine") so runs with different settings
# get their own folder instead of resuming each other's checkpoints. "" = none.
EXP_TAG = ""


def exp_name(test_fold):
    """Run folder name for one fold, e.g. "diffusion_avg_eq_paired_1234" (mirrors the prior naming)."""
    train_folds = "".join(str(f) for f in range(1, NUM_FOLDS + 1) if f != test_fold)
    return f"diffusion_{DATASET_NAME}_{train_folds}" + (f"_{EXP_TAG}" if EXP_TAG else "")


EXP_NAME = exp_name(TEST_FOLD)
RUN_DIR = OUTPUT_ROOT / EXP_NAME
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
SAMPLE_DIR = RUN_DIR / "samples"
RESULTS_DIR = RUN_DIR / "results"


def as_dict():
    """Snapshot of all upper-case settings, saved alongside each run."""
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in globals().items()
        if k.isupper()
    }
