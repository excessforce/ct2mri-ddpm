"""Average results over the cross-validation folds.

    python aggregate.py runs/ddpm_ct2mri --sampler ddpm
"""
import argparse
import glob
import json
import os

import numpy as np

METRICS = ["fid", "mse", "mae", "psnr", "ssim", "cos_sim"]

ap = argparse.ArgumentParser()
ap.add_argument("root", help="e.g. runs/ddpm_ct2mri")
ap.add_argument("--sampler", default="ddpm")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.root, "fold*", f"generated_{a.sampler}", "results.json")))
if not files:
    raise SystemExit(f"no results.json found under {a.root}/fold*/generated_{a.sampler}/")
rows = []
for f in files:
    with open(f) as fh:
        rows.append(json.load(fh))

summary = {}
print(f"{len(rows)} folds: " + ", ".join(os.path.basename(os.path.dirname(os.path.dirname(f))) for f in files))
for k in METRICS:
    vals = [r[k] for r in rows if k in r]
    if vals:
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": len(vals)}
        print(f"{k:>8}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")

with open(os.path.join(a.root, f"summary_{a.sampler}.json"), "w") as fh:
    json.dump(summary, fh, indent=2)