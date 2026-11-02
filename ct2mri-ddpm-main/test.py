"""
Evaluate a trained diffusion model on the held-out fold.

    python test.py

Loads the EMA weights from runs/<EXP_NAME>/checkpoints/<CHECKPOINT_NAME>,
generates an sMRI for every slice of TEST_FOLD and writes to runs/<EXP_NAME>/results/:
    results_<timestamp>.log        prior-work format: "fid, mse, mae, cs, psnr,ssim"
                                   (SSIM column uses the legacy definition, see metrics.py)
    metrics_summary_<timestamp>.csv  mean and std of every metric, both SSIM variants
    per_image_<timestamp>.csv        metrics for each slice
    predictions_fold<k>.npz          arr_0 CT, arr_1 rMRI, arr_2 sMRI, (N, 256, 256, 1) in [-1, 1]
    figures/                         CT | rMRI | sMRI plots for the first NUM_TEST_FIGURES slices
"""

import csv
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import config
import utils
from data_loader import CTMRIDataset, build_condition, get_test_loader
from metrics import FIDCalculator, image_metrics
from unet import build_unet

METRIC_NAMES = ["mse", "mae", "cs", "psnr", "ssim", "ssim_legacy"]


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    utils.make_dirs()
    figure_dir = config.RESULTS_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)

    ckpt_path = config.CHECKPOINT_DIR / config.CHECKPOINT_NAME
    ckpt = utils.load_checkpoint(ckpt_path, device)
    model = build_unet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded EMA weights from {ckpt_path} (epoch {ckpt['epoch']}, step {ckpt['step']})")

    scheduler = utils.build_scheduler()
    test_set = CTMRIDataset([config.TEST_FOLD], augment=False)
    loader = get_test_loader(test_set)
    print(f"Test fold {config.TEST_FOLD}: {len(test_set)} slices | "
          f"{config.SAMPLER} sampling, {config.SAMPLING_STEPS} steps")

    fid = FIDCalculator(device) if config.COMPUTE_FID else None
    generator = torch.Generator(device=device).manual_seed(config.SEED)
    rows, cts, reals, fakes = [], [], [], []
    start = time.time()

    for batch in tqdm(loader, desc="test batches"):
        cond = build_condition(batch).to(device)
        real = batch["mri"].to(device)
        with utils.autocast(device):
            fake = scheduler.sample(model, cond, config.SAMPLER, config.SAMPLING_STEPS,
                                    config.DDIM_ETA, generator, config.IMG_CHANNELS)

        batch_metrics = image_metrics(real, fake)
        for i, idx in enumerate(batch["index"].tolist()):
            rows.append({"index": idx, **{k: batch_metrics[k][i].item() for k in METRIC_NAMES}})
            if idx < config.NUM_TEST_FIGURES:
                utils.save_comparison(figure_dir / f"{config.EXP_NAME}_plot_{idx}.png",
                                      batch["ct"][i:i + 1], real[i:i + 1], fake[i:i + 1])
        if fid is not None:
            fid.update(real, fake)
        if config.SAVE_PREDICTIONS:
            cts.append(batch["ct"])
            reals.append(real.cpu())
            fakes.append(fake.cpu())

    duration = time.time() - start
    fid_value = fid.compute() if fid is not None else float("nan")
    means = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_NAMES}
    stds = {k: float(np.std([r[k] for r in rows])) for k in METRIC_NAMES}

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out = config.RESULTS_DIR

    # Same layout as the prior work's results_*.log so existing aggregation still parses it.
    legacy = [fid_value, means["mse"], means["mae"], means["cs"], means["psnr"], means["ssim_legacy"]]
    np.savetxt(out / f"results_{stamp}.log", [legacy], fmt="%.6f",
               header="fid, mse, mae, cs, psnr,ssim", delimiter=",")

    summary = [{"metric": "fid", "mean": fid_value, "std": ""}]
    summary += [{"metric": k, "mean": means[k], "std": stds[k]} for k in METRIC_NAMES]
    summary.append({"metric": "sampling_seconds", "mean": duration, "std": ""})
    write_csv(out / f"metrics_summary_{stamp}.csv", ["metric", "mean", "std"], summary)
    write_csv(out / f"per_image_{stamp}.csv", ["index"] + METRIC_NAMES, rows)

    if config.SAVE_PREDICTIONS:
        def nhwc(tensors):
            return torch.cat(tensors).float().permute(0, 2, 3, 1).numpy()
        np.savez_compressed(out / f"predictions_fold{config.TEST_FOLD}.npz",
                            nhwc(cts), nhwc(reals), nhwc(fakes))

    print(f"\nResults for {config.EXP_NAME} (test fold {config.TEST_FOLD}, {len(rows)} slices)")
    print(f"  {'fid':<12}{fid_value:.4f}")
    for k in METRIC_NAMES:
        print(f"  {k:<12}{means[k]:.4f} +/- {stds[k]:.4f}")
    print(f"Sampling time: {duration:.1f} s | results written to {out}")


if __name__ == "__main__":
    main()
