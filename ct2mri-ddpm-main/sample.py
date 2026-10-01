"""Generate MRI for a held-out CV fold, or a separate test set, from a trained run and score it.

    # held-out CV fold
    python sample.py --run_dir runs/ddpm_ct2mri/fold5 --data_path /path/to/prefix_fold --fold 5
    # separate test set: one .npz with arr_0 = CT, arr_1 = MRI
    python sample.py --run_dir runs/ddpm_ct2mri/fold5 --data_file /path/to/avg_eq_seg_test.npz
    # quick check with fewer steps (use the default ddpm sampler for final numbers)
    python sample.py ... --sampler ddim --ddim_steps 100

Saves to <run_dir>/generated_<sampler>[_<test file name>]/ : pred.npy, gt.npy, ct.npy
(N,H,W,1 in [-1,1], the old TF layout), preview.png, results.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Subset

from dataset import PairedSliceDataset, load_npz, make_loader
from ddpm_main import GaussianDiffusion, UNet
from metrics import evaluate
from train import save_preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_path", help="fold file prefix (use together with --fold)")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--data_file", help="single .npz test set (instead of --data_path/--fold)")
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--weights", choices=["ema", "ddpm"], default="ema")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N slices")
    ap.add_argument("--input_range", type=float, nargs=2, default=None,
                    help="value range of the .npz; default = the range used for training")
    ap.add_argument("--legacy_ssim", action="store_true")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if bool(a.data_file) == bool(a.data_path and a.fold is not None):
        ap.error("give either --data_file, or --data_path together with --fold")

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(a.run_dir, "ckpt.pt"), map_location=dev)
    ta = ck["args"]
    ddpm = GaussianDiffusion(UNet(base=ta["base_channels"]), ta["timesteps"])
    ddpm.load_state_dict(ck[a.weights])
    ddpm.to(dev).eval()
    print(f"loaded {a.weights} weights from epoch {ck['epoch']}")

    if a.data_file:
        src = a.data_file if a.data_file.endswith(".npz") else a.data_file + ".npz"
        tag = "_" + os.path.splitext(os.path.basename(src))[0]
    else:
        src, tag = f"{a.data_path}{a.fold}.npz", ""
    ct, mri, _ = load_npz([src])
    rng = tuple(a.input_range or ta.get("input_range", [-1.0, 1.0]))
    ds = PairedSliceDataset(ct, mri, input_range=rng, image_size=ta["image_size"])
    if a.limit:
        ds = Subset(ds, range(min(a.limit, len(ds))))
    loader = make_loader(ds, a.batch_size, train=False, num_workers=2)
    print(f"{len(ds)} slices from {src}")

    cts, gts, preds = [], [], []
    for i, batch in enumerate(loader):
        c = batch["ct"].to(dev)
        p = ddpm.sample(c) if a.sampler == "ddpm" else ddpm.ddim_sample(c, a.ddim_steps)
        cts.append(batch["ct"]); gts.append(batch["mri"]); preds.append(p.cpu())
        print(f"batch {i + 1}/{len(loader)}")
    cts, gts, preds = torch.cat(cts), torch.cat(gts), torch.cat(preds)

    out = os.path.join(a.run_dir, f"generated_{a.sampler}{tag}")
    os.makedirs(out, exist_ok=True)
    for name, x in (("pred", preds), ("gt", gts), ("ct", cts)):
        np.save(os.path.join(out, f"{name}.npy"), x.permute(0, 2, 3, 1).numpy())
    save_preview(os.path.join(out, "preview.png"), cts[:8], gts[:8], preds[:8])

    res = evaluate(preds, gts, fid=not a.no_fid, legacy_ssim=a.legacy_ssim)
    res.update(sampler=a.sampler, ddim_steps=a.ddim_steps if a.sampler == "ddim" else None,
               weights=a.weights, epoch=ck["epoch"], source=src)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()