"""Metrics for CT -> MRI synthesis. They score saved arrays, so GAN and DDPM outputs can be
evaluated with exactly the same code.

    python metrics.py --pred pred.npy --target gt.npy [--input_range -1 1] [--no_fid]

Arrays: (N, H, W) or (N, H, W, 1). Images are rescaled to [-1, 1] internally.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from dataset import _to_nchw


def _gauss_window(size, sigma, device):
    c = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, data_range=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """Per-image SSIM, same settings as tf.image.ssim (gaussian 11x11, sigma 1.5, valid conv)."""
    w = _gauss_window(size, sigma, x.device)
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x ** 2
    syy = F.conv2d(y * y, w) - mu_y ** 2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def image_metrics(pred, target, legacy_ssim=False):
    """pred, target: (B, 1, H, W) in [-1, 1]. Returns per-image tensors, computed on [0, 1] images."""
    p, t = ((pred + 1) / 2).clamp(0, 1), ((target + 1) / 2).clamp(0, 1)
    mse = ((p - t) ** 2).mean(dim=(1, 2, 3))
    out = {
        "mse": mse,
        "mae": (p - t).abs().mean(dim=(1, 2, 3)),
        "psnr": 10 * torch.log10(1.0 / mse.clamp_min(1e-12)),
        "cos_sim": F.cosine_similarity(p.flatten(1), t.flatten(1), dim=1),
    }
    if legacy_ssim:  # old evaluate.py rescaled already-[0,1] images a second time before SSIM
        out["ssim"] = ssim((p + 1) / 2, (t + 1) / 2)
    else:
        out["ssim"] = ssim(p, t)
    return out


def fid_score(pred, target, device="cpu", batch=64):
    """FID over the WHOLE set (not per batch)."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for i in range(0, len(pred), batch):
        for imgs, real in ((target, True), (pred, False)):
            x = ((imgs[i:i + batch] + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1).to(device)
            fid.update(x, real=real)
    return float(fid.compute())


@torch.no_grad()
def evaluate(pred, target, fid=True, legacy_ssim=False, device=None):
    """pred, target: (N, 1, H, W) tensors in [-1, 1]. Returns a dict of means."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    per = {}
    for i in range(0, len(pred), 32):
        m = image_metrics(pred[i:i + 32].to(device), target[i:i + 32].to(device), legacy_ssim)
        for k, v in m.items():
            per.setdefault(k, []).append(v.cpu())
    res = {k: float(torch.cat(v).mean()) for k, v in per.items()}
    if fid:
        try:
            res["fid"] = fid_score(pred, target, device)
        except ImportError:
            print('FID skipped: pip install "torchmetrics[image]"')
        except Exception as e:  # e.g. a compute node with no internet to fetch the Inception weights
            print(f"FID skipped: {type(e).__name__}: {e}")
    res["n"] = len(pred)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--input_range", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--legacy_ssim", action="store_true", help="reproduce the old double-rescaled SSIM")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--out", default=None, help="optional json path")
    a = ap.parse_args()
    rng = tuple(a.input_range)
    pred = _to_nchw(np.load(a.pred), "pred", rng)
    target = _to_nchw(np.load(a.target), "target", rng)
    res = evaluate(pred, target, fid=not a.no_fid, legacy_ssim=a.legacy_ssim)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)