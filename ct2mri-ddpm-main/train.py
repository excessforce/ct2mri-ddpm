"""
Train the conditional diffusion model (CT -> MRI).

    python train.py

All settings come from config.py. Outputs go to runs/<EXP_NAME>/:
    checkpoints/last.pt   overwritten every CKPT_EVERY epochs (used to resume)
    checkpoints/best.pt   lowest validation MAE on the sampled validation subset
    checkpoints/final.pt  weights after the last epoch
    samples/              CT | rMRI | sMRI figures from each validation run
    train_log.csv         per-epoch losses and validation metrics
"""

import csv
import math
import time

import torch

import config
import utils
from data_loader import build_condition, get_dataloaders
from losses import DiffusionLoss
from metrics import image_metrics
from unet import build_unet


def lr_lambda(total_steps):
    def fn(step):
        if step < config.WARMUP_STEPS:
            return (step + 1) / config.WARMUP_STEPS
        if config.LR_SCHEDULE == "cosine":
            progress = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return 1.0
    return fn


def diffusion_step(model, batch, scheduler, loss_fn, device, generator=None):
    """Noise the MRI at random timesteps and return the denoising loss."""
    x0 = batch["mri"].to(device, non_blocking=True)
    cond = build_condition(batch).to(device, non_blocking=True)

    t = scheduler.sample_timesteps(x0.shape[0], device, generator)
    noise = torch.randn(x0.shape, device=device, generator=generator)
    x_t = scheduler.add_noise(x0, noise, t)

    with utils.autocast(device):
        pred = model(torch.cat([x_t, cond], dim=1), t)
    return loss_fn(pred, scheduler.get_target(x0, noise, t), t)


@torch.no_grad()
def validation_loss(model, loader, scheduler, loss_fn, device):
    """Denoising loss on the test fold with fixed timesteps/noise so epochs are comparable."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.SEED)
    total, count = 0.0, 0
    for batch in loader:
        n = batch["mri"].shape[0]
        total += diffusion_step(model, batch, scheduler, loss_fn, device, generator).item() * n
        count += n
    model.train()
    return total / count


@torch.no_grad()
def validation_samples(ema_model, dataset, scheduler, device, epoch):
    """Sample a fixed, evenly spaced subset of the test fold and report pixel metrics."""
    n = min(config.VAL_NUM_SAMPLES, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, n).long().tolist()
    batch = torch.utils.data.default_collate([dataset[i] for i in indices])
    cond = build_condition(batch).to(device)
    real = batch["mri"].to(device)

    generator = torch.Generator(device=device).manual_seed(config.SEED)
    fakes = []
    with utils.autocast(device):
        for chunk in cond.split(config.EVAL_BATCH_SIZE):
            fakes.append(scheduler.sample(ema_model, chunk, config.SAMPLER, config.SAMPLING_STEPS,
                                          config.DDIM_ETA, generator, config.IMG_CHANNELS))
    fake = torch.cat(fakes)

    k = config.NUM_PLOT_SAMPLES
    utils.save_comparison(config.SAMPLE_DIR / f"epoch_{epoch:04d}.png",
                          batch["ct"][:k], real[:k], fake[:k])
    return {name: values.mean().item() for name, values in image_metrics(real, fake).items()}


def main():
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    utils.make_dirs()
    utils.save_config_snapshot(config.RUN_DIR / "config.json")
    print(f"Experiment: {config.EXP_NAME} | device: {device}")
    print(f"Data: {config.DATA_PATH_PREFIX}[{''.join(map(str, config.TRAIN_FOLDS))}].npz, "
          f"test fold {config.TEST_FOLD}")

    train_loader, test_loader = get_dataloaders()
    print(f"Train slices: {len(train_loader.dataset)} | test slices: {len(test_loader.dataset)}")

    scheduler = utils.build_scheduler()
    loss_fn = DiffusionLoss(scheduler, config.LOSS_TYPE, config.MIN_SNR_GAMMA)

    model = build_unet().to(device)
    ema = utils.EMA(model, config.EMA_DECAY)
    print(f"UNet parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    total_steps = config.EPOCHS * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps))

    start_epoch, step, best_mae = 0, 0, float("inf")
    last_path = config.CHECKPOINT_DIR / "last.pt"
    if config.RESUME and last_path.exists():
        ckpt = utils.load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch, step, best_mae = ckpt["epoch"] + 1, ckpt["step"], ckpt["best_metric"]
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    train_model = torch.compile(model) if config.USE_COMPILE else model

    log_path = config.RUN_DIR / "train_log.csv"
    log_fields = ["epoch", "step", "lr", "train_loss", "val_loss",
                  "val_mse", "val_mae", "val_cs", "val_psnr", "val_ssim", "val_ssim_legacy", "seconds"]
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, log_fields).writeheader()

    train_start = time.time()
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()
        train_model.train()
        running, count = 0.0, 0

        for batch in train_loader:
            loss = diffusion_step(train_model, batch, scheduler, loss_fn, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()
            ema.update(model)

            running += loss.item()
            count += 1
            step += 1

        row = {
            "epoch": epoch,
            "step": step,
            "lr": lr_scheduler.get_last_lr()[0],
            "train_loss": running / count,
            "val_loss": validation_loss(train_model, test_loader, scheduler, loss_fn, device),
        }

        last_epoch = epoch == config.EPOCHS - 1
        if (epoch + 1) % config.VAL_EVERY == 0 or last_epoch:
            val = validation_samples(ema.model, test_loader.dataset, scheduler, device, epoch)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["mae"] < best_mae:
                best_mae = val["mae"]
                utils.save_checkpoint(config.CHECKPOINT_DIR / "best.pt", model, ema, optimizer,
                                      lr_scheduler, epoch, step, best_mae)

        row["seconds"] = time.time() - epoch_start
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)
        print(" | ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

        if (epoch + 1) % config.CKPT_EVERY == 0 or last_epoch:
            utils.save_checkpoint(last_path, model, ema, optimizer, lr_scheduler, epoch, step, best_mae)

    utils.save_checkpoint(config.CHECKPOINT_DIR / "final.pt", model, ema, optimizer, lr_scheduler,
                          config.EPOCHS - 1, step, best_mae)
    duration = time.time() - train_start
    print(f"Training time: {duration:.2f} seconds")
    (config.RUN_DIR / "train_time.txt").write_text(f"Training time: {duration:.2f} seconds\n")


if __name__ == "__main__":
    main()
