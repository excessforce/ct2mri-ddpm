# Changelog

## 2026-11-02: Cleanup and test-run preparation

Verified with `python -m pytest -q` on CPU (Python 3.14, torch 2.14): **17 passed** in 2 to 4 minutes.

### Removed (redundant)

- **`ct2mri-ddpm-main/`** (git root): the earlier argparse prototype (`dataset.py`, `ddpm_main.py`,
  `train.py`, `sample.py`, `metrics.py`, `aggregate.py`, `run_folds.py`). It duplicated every module of
  this project and broke the ground rules in `CLAUDE.md`: argparse instead of `config.py`, `CUDA_VISIBLE_DEVICES`
  in `run_folds.py`, 512-only "bad image" lists, a `torchmetrics` dependency and a second UNet/scheduler.
  Its only feature without an equivalent here (fold aggregation) has been re-implemented as `aggregate.py`.
  The code is still in git history (commit `070a8bd`).
- `.gitignore` (this folder): dropped `__pycache__/` and `*.pyc`, which the repository-root `.gitignore`
  already covers.
- `train.py`: the inline `torch.randint` timestep draw duplicated `NoiseScheduler.sample_timesteps`,
  which was unused. Training now calls the scheduler method (it gained a `generator` argument, so
  validation noise stays fixed).
- Stale `__pycache__/` folders.

### Fixed

- `data_loader.load_folds`: the `use_mask` default was bound to `config.USE_MASK` at import time,
  so patching config (tests, scripts) had no effect. It is now read at call time, as `CLAUDE.md` §10 requires.
- `data_loader`: `pin_memory` is only enabled when CUDA is available (removes the warning on CPU runs).
- `utils.save_checkpoint`: writes to `<name>.tmp` and then renames, so a job killed mid-save can no
  longer leave a truncated `last.pt` that breaks `RESUME`.
- `utils.get_device`: a clear error when `DEVICE` names a GPU index that doesn't exist (e.g. `cuda:1`
  on a single-GPU machine), instead of an opaque CUDA error.

### Added

- `config.EXP_TAG` and `config.exp_name(test_fold)`: an optional run-folder suffix for ablations, so
  different settings no longer resume each other's checkpoints (`CLAUDE.md` §7, former item 7).
- `aggregate.py`: averages the newest `metrics_summary_*.csv` of each fold and writes
  `runs/diffusion_<dataset>[_<tag>]_cv_summary.csv` with mean, std and per-fold values (former item 3).
- `tests/` + `pytest.ini` (former item 9):
  - `test_scheduler.py`: oracle reconstruction for linear/cosine × eps/v × ddim/ddpm, and a v-target round trip.
  - `test_metrics.py`: perfect scores for identical images, known MSE/MAE/PSNR, analytic Fréchet distances.
  - `test_aggregate.py`: newest-summary selection, mean/std across folds, error when nothing exists.
  - `test_smoke.py`: synthetic 5-fold data, tiny CPU UNet, `train.main()` → resume → `test.py`, for paired
    and masked data; plus the full-size UNet forward pass at 256×256.
- `requirements.txt`: `pytest`.
- `.gitignore`: `*.pt` and `*.npz`, so checkpoints, predictions and data never get committed.

### Documentation

- `CLAUDE.md`: layout table, `EXP_TAG`, aggregation, the pytest-based smoke test (§3.8), the clone path
  (the project root is the `ct2mri-ddpm/` sub-folder), and the status list (§7) updated.
- `README.md`: new files, `EXP_TAG`, and the test / aggregate commands.

### Not changed

- Model, scheduler, loss and metric definitions are untouched, so results remain comparable with GAN-CM.
- The repository-root `README.md` shows as deleted in the working tree; that deletion predates this
  change set and was left as is.
