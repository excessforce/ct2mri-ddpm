"""Train the conditional DDPM (CT -> MRI) on one cross-validation fold.

    python train.py --data_path /path/to/prefix_fold --fold 5 --exp_name ddpm_ct2mri

--data_path is the file prefix: files are {data_path}{i}.npz (same as the old flags.data_path).
Outputs go to runs/<exp_name>/fold<k>/ : args.json, log.csv, ckpt.pt, samples/
"""
import argparse
import contextlib
import copy
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import build_fold_datasets, make_loader
from ddpm_main import GaussianDiffusion, UNet


def get_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", required=True, help="fold file prefix; files are {prefix}{i}.npz")
    p.add_argument("--fold", type=int, default=5, help="held-out fold (the old --test_fold)")
    p.add_argument("--exp_name", default="ddpm_ct2mri")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--input_range", type=float, nargs=2, default=[-1.0, 1.0],
                   help="value range stored in the .npz files")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--aug", action="store_true", help="random flips / rot90 (off by default)")
    # model / diffusion
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=1000)
    # optimisation
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # monitoring
    p.add_argument("--sample_every", type=int, default=10, help="epochs between preview grids")
    p.add_argument("--save_every", type=int, default=10, help="epochs between checkpoints")
    p.add_argument("--preview_steps", type=int, default=50, help="DDIM steps for previews")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def save_preview(path, ct, gt, pred):
    """Rows = samples, columns = CT | real MRI | synthetic MRI. Inputs (B,1,H,W) in [-1,1]."""
    rows = [torch.cat([c, g, p], dim=-1) for c, g, p in zip(ct, gt, pred)]
    grid = torch.cat(rows, dim=-2)[0].cpu().numpy()
    plt.imsave(path, (grid + 1) / 2, cmap="gray", vmin=0, vmax=1)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.join(args.out_dir, args.exp_name, f"fold{args.fold}")
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    aug = {} if args.aug else dict(p_flip_h=0.0, p_flip_v=0.0, p_rot90=0.0)
    train_ds, _, test_ds = build_fold_datasets(
        args.data_path, args.fold, input_range=tuple(args.input_range),
        image_size=args.image_size, **aug)
    loader = make_loader(train_ds, args.batch_size, train=True, num_workers=args.num_workers)
    print(f"train slices: {len(train_ds)} | held-out slices: {len(test_ds)} | device: {dev}")

    ddpm = GaussianDiffusion(UNet(base=args.base_channels), args.timesteps).to(dev)
    ema = copy.deepcopy(ddpm).eval().requires_grad_(False)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"parameters: {sum(p.numel() for p in ddpm.parameters()) / 1e6:.1f}M")

    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    start = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        ddpm.load_state_dict(ck["ddpm"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"]
        print(f"resumed from epoch {start}")

    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "seconds"])

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.amp and dev == "cuda" else contextlib.nullcontext())

    for epoch in range(start, args.epochs):
        ddpm.train()
        t0, total, n = time.time(), 0.0, 0
        for batch in loader:
            ct = batch["ct"].to(dev, non_blocking=True)
            mri = batch["mri"].to(dev, non_blocking=True)
            with amp:
                loss = ddpm.loss(mri, ct)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            ema_update(ema, ddpm, args.ema_decay)
            total, n = total + loss.item(), n + 1
        avg = total / max(n, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {avg:.5f}  {time.time() - t0:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg:.6f}", f"{time.time() - t0:.1f}"])

        if (epoch + 1) % args.sample_every == 0:
            k = min(4, len(test_ds))
            ct = torch.stack([test_ds[i]["ct"] for i in range(k)]).to(dev)
            gt = torch.stack([test_ds[i]["mri"] for i in range(k)])
            pred = ema.ddim_sample(ct, steps=args.preview_steps).cpu()
            save_preview(os.path.join(run_dir, "samples", f"epoch{epoch + 1:04d}.png"),
                         ct.cpu(), gt, pred)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            tmp = ckpt_path + ".tmp"  # write-then-rename so a preempted job never leaves a half file
            torch.save({"ddpm": ddpm.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "epoch": epoch + 1, "args": vars(args)}, tmp)
            os.replace(tmp, ckpt_path)


if __name__ == "__main__":
    main()