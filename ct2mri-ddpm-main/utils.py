"""Shared helpers for training and testing."""

import contextlib
import copy
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from noise_scheduler import NoiseScheduler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if config.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable, falling back to CPU instead of {config.DEVICE}")
        return torch.device("cpu")
    device = torch.device(config.DEVICE)
    if device.type == "cuda":
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"DEVICE = {config.DEVICE!r} but only {torch.cuda.device_count()} "
                               "CUDA device(s) are visible; update DEVICE in config.py")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device


def autocast(device):
    if config.USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_scheduler():
    return NoiseScheduler(
        num_timesteps=config.NUM_TRAIN_TIMESTEPS,
        beta_schedule=config.BETA_SCHEDULE,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        prediction_type=config.PREDICTION_TYPE,
    )


def make_dirs():
    for d in (config.CHECKPOINT_DIR, config.SAMPLE_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_config_snapshot(path):
    with open(path, "w") as f:
        json.dump(config.as_dict(), f, indent=2, default=str)


class EMA:
    """Exponential moving average of model weights, used for sampling."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


def unwrap(model):
    """Return the underlying module of a torch.compile'd model."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path, model, ema, optimizer, lr_scheduler, epoch, step, best_metric):
    # Write then rename, so a job killed mid-save never leaves a truncated checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": unwrap(model).state_dict(),
        "ema": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config.as_dict(),
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def save_comparison(path, ct, real, fake, title="Diffusion sMRI"):
    """One row per example: CT | rMRI | sMRI (all inputs in [-1, 1])."""
    n = ct.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    for i in range(n):
        for ax, img, name in zip(axes[i], (ct[i], real[i], fake[i]), ("CT", "rMRI", title)):
            ax.imshow((img.squeeze().float().cpu().numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.set_title(name, fontsize=16)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
