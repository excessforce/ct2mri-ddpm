"""
End-to-end smoke test on synthetic data: train (with resume) then test, on CPU,
with a tiny UNet. Checks that every output file is written.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import config
import train
from unet import build_unet

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_test_script():
    # Loaded by path: "import test" could resolve to Python's stdlib test package.
    spec = importlib.util.spec_from_file_location("ct2mri_test", PROJECT_DIR / "test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_synthetic_folds(data_dir, with_mask, slices=6):
    rng = np.random.default_rng(0)
    shape = (slices, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
    for fold in range(1, config.NUM_FOLDS + 1):
        arrays = [rng.uniform(-1, 1, shape).astype(np.float32) for _ in range(2)]
        if with_mask:
            arrays.append((rng.uniform(0, 1, shape) > 0.5).astype(np.float32))
        np.savez(data_dir / f"fold{fold}.npz", *arrays)


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    settings = {
        "DATA_PATH_PREFIX": str(tmp_path / "fold"),
        "RUN_DIR": run_dir,
        "CHECKPOINT_DIR": run_dir / "checkpoints",
        "SAMPLE_DIR": run_dir / "samples",
        "RESULTS_DIR": run_dir / "results",
        "DEVICE": "cpu",
        "NUM_WORKERS": 0,
        "BASE_CHANNELS": 16,
        "GROUP_NORM_GROUPS": 8,
        "CHANNEL_MULT": (1, 1, 1, 1, 1, 1),
        "NUM_RES_BLOCKS": 1,
        "ATTENTION_RESOLUTIONS": (8,),
        "EPOCHS": 2,
        "BATCH_SIZE": 4,
        "EVAL_BATCH_SIZE": 4,
        "WARMUP_STEPS": 2,
        "CKPT_EVERY": 1,
        "VAL_EVERY": 1,
        "VAL_NUM_SAMPLES": 2,
        "NUM_PLOT_SAMPLES": 2,
        "SAMPLING_STEPS": 3,
        "COMPUTE_FID": False,
        "NUM_TEST_FIGURES": 2,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    return tmp_path


@pytest.mark.parametrize("with_mask", [False, True], ids=["paired", "masked"])
def test_train_resume_test(tiny_config, monkeypatch, with_mask):
    monkeypatch.setattr(config, "USE_MASK", with_mask)
    monkeypatch.setattr(config, "COND_CHANNELS", 2 if with_mask else 1)
    write_synthetic_folds(tiny_config, with_mask)

    train.main()
    monkeypatch.setattr(config, "EPOCHS", 3)
    train.main()    # resumes from last.pt and trains one more epoch

    with open(config.RUN_DIR / "train_log.csv", newline="") as f:
        assert [int(r["epoch"]) for r in csv.DictReader(f)] == [0, 1, 2]
    for name in ("last.pt", "best.pt", "final.pt"):
        assert (config.CHECKPOINT_DIR / name).exists()
    assert not list(config.CHECKPOINT_DIR.glob("*.tmp"))
    assert (config.SAMPLE_DIR / "epoch_0002.png").exists()

    load_test_script().main()
    results = config.RESULTS_DIR
    for pattern in ("results_*.log", "metrics_summary_*.csv", "per_image_*.csv", "figures/*.png"):
        assert list(results.glob(pattern)), pattern
    with np.load(results / f"predictions_fold{config.TEST_FOLD}.npz") as preds:
        assert preds["arr_2"].shape == (6, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        assert np.abs(preds["arr_2"]).max() <= 1.0


def test_default_unet_shapes():
    model = build_unet()
    x = torch.randn(1, config.IMG_CHANNELS + config.COND_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(x, torch.tensor([500]))
    assert out.shape == (1, config.IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
