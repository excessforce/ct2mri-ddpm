# CLAUDE.md

Guidance for Claude Code (and any collaborator) working in this repository.

This project is a **PyTorch conditional diffusion model that translates 256×256 CT
slices into MRI**. It follows up on the earlier TensorFlow GAN-CM project ("gan-cm").
It uses the same dataset and reports the same metrics so the two can be compared,
but it is a separate codebase and shares no code with it.

---

## 1. Ground rules (from the project owner, do not break these)

1. **PyTorch only.** No TensorFlow or Keras anywhere in this repo.
2. **Independent of the old code.** The GAN-CM repository may be *read* for context
   (data format, preprocessing, metric definitions), but nothing from it may be
   imported, vendored, copied in as a module, or called at runtime. Re-implement
   what's needed here.
3. **Diffusion, not GAN.** The model is a denoising diffusion model (UNet + noise
   scheduler). Don't add adversarial losses or discriminators.
4. **256×256 only.** Don't add 512×512 support. Diffusion at 512 costs too much
   compute. `IMAGE_SIZE` must stay 256.
5. **Same dataset, comparable metrics.** Use the same five-fold `.npz` data and report
   FID, MSE, MAE, cosine similarity, PSNR and SSIM, with the same definitions as the
   earlier work (see §5). Don't change how a metric is computed without keeping a
   column that is still comparable with the old results.
6. **Configuration is constants in `config.py`, not argparse.** Add new settings as
   upper-case constants in `config.py`. The only runtime override is the
   `CT2MRI_DATA_ROOT` environment variable for the data location.
7. **Data lives on the remote server** (reached over SSH). The path is a constant in
   `config.py` (`DATA_ROOT`). Never hard-code data paths elsewhere.
8. **GPU: physical GPU 0 on the server is `cuda:1` in PyTorch.** `DEVICE = "cuda:1"`.
   Don't set `CUDA_VISIBLE_DEVICES`, because that renumbers the devices and breaks the mapping.
9. Keep the code modular and free of the copy-paste duplication that made the old
   repo hard to maintain. There should be one data loader, one scheduler, one UNet,
   one training script and one test script. Variants go behind config flags, not
   copied files.

---

## 2. Repository layout

| File | Purpose |
| --- | --- |
| `config.py` | Every setting: data path, folds, GPU, schedule, UNet size, loss, training, testing, output paths. Also `as_dict()` for config snapshots. |
| `data_loader.py` | `CTMRIDataset` (loads folds, NHWC to NCHW, paired augmentation), `build_condition()`, loader factories. Run directly to sanity-check the data. |
| `noise_scheduler.py` | `NoiseScheduler`: linear or cosine betas, `add_noise`, ε/v targets, `predict_x0_and_eps`, DDPM and DDIM steps, and the `sample()` loop. |
| `timestep_encoder.py` | Sinusoidal embedding plus a 2-layer MLP. |
| `unet.py` | Conditional UNet (ResBlocks with timestep scale/shift, multi-head self-attention at 16×16 and 8×8 plus the bottleneck, zero-initialized residual outputs). `build_unet()` builds it from config. |
| `losses.py` | `DiffusionLoss`: MSE, L1 or Huber on the ε/v target, optional Min-SNR-γ weighting. |
| `metrics.py` | Per-image MSE, MAE, CS, PSNR, SSIM, `ssim_legacy`; `FIDCalculator` (torchvision InceptionV3); `frechet_distance` (no SciPy). |
| `train.py` | Training loop: AdamW, warmup, EMA, bf16 autocast, gradient clipping, resume, per-epoch validation loss, periodic sampled validation, checkpoints, CSV log. |
| `test.py` | Samples every slice of the held-out fold with the EMA weights and writes logs, CSVs, figures and a predictions `.npz`. |
| `aggregate.py` | Averages the newest `metrics_summary_*.csv` of each fold into a cross-validation summary (mean ± std across folds). |
| `utils.py` | Seeding, device setup, autocast, scheduler factory, EMA, atomic checkpoint save/load, comparison figures. |
| `tests/` | pytest suite: oracle scheduler test, metric sanity checks, aggregation, and an end-to-end train/resume/test smoke test on synthetic data. |
| `pytest.ini` | Points pytest at `tests/` (so it never touches `test.py`). |
| `CHANGELOG.md` | Log of changes to the codebase. |
| `README.md` | User-facing summary. |
| `runs/` | All outputs (git-ignored). |

Modules import each other with flat imports (`import config`, `from unet import build_unet`).
Run every script from the repository root.

---

## 3. How to run it

### 3.1 Get onto the server and set up the environment

```bash
ssh <user>@<server>                     # the aisec server that holds /media/aisec-102/DATA3
git clone <new-repo-url> ct2mri_diffusion
cd ct2mri_diffusion/ct2mri-ddpm          # the project root is this sub-folder of the git repo

conda create -n ct2mri python=3.11 -y
conda activate ct2mri
# Install a CUDA build of torch that matches the server's driver (check `nvidia-smi`),
# e.g. for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Python 3.10 or newer and torch 2.2 or newer are required (the code uses
`F.scaled_dot_product_attention` and `torch.load(weights_only=...)`).

### 3.2 Check the GPU mapping

```bash
nvidia-smi                               # note which physical GPU is "GPU 0" and that it is free
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

PyTorch orders GPUs fastest-first by default, while `nvidia-smi` orders them by PCI bus.
That's why physical GPU 0 shows up as `cuda:1`. Confirm that the name printed for index 1
matches the GPU you intend to use. If the server's GPUs change, update `DEVICE` in `config.py`.

### 3.3 Point at the data and check it

The default `DATA_ROOT` is `/media/aisec-102/DATA3/rachel/data/CV`. Files are read from:

```
{DATA_ROOT}/{EQUALIZATION}_{MASK_TYPE}/{FILE_PREFIX}{fold}.npz
  paired:  .../avg_eq_paired/norm_neg1pos1_fold1.npz ... fold5.npz
  masked:  .../avg_eq_seg/norm_mask_neg1pos1_fold1.npz ... fold5.npz
```

```bash
ls /media/aisec-102/DATA3/rachel/data/CV/          # confirm the real folder names first (see §8)
python data_loader.py                              # prints slices and value ranges per fold
```

Expected: CT and MRI ranges close to [-1, 1], and a few hundred to a few thousand slices per fold.
If the data is mounted somewhere else (for example `sshfs` from a workstation):

```bash
export CT2MRI_DATA_ROOT=/mnt/aisec/rachel/data/CV
```

### 3.4 Configure an experiment (`config.py`)

Typically you only change:

- `EQUALIZATION`: `"avg_eq"`, `"eq"` or `"no_eq"`
- `MASK_TYPE`: `"paired"` (CT only) or `"seg"` / `"bin_edge"` (CT + mask). `USE_MASK`,
  `FILE_PREFIX` and `COND_CHANNELS` follow automatically.
- `TEST_FOLD`: 1 to 5
- `BATCH_SIZE`, `EPOCHS`, and the diffusion options if you're running an ablation
- `EXP_TAG`: a short suffix for ablations (e.g. `"v_cosine"`), so a run with different settings
  gets its own folder instead of resuming another run's checkpoint

The run directory is `runs/diffusion_{EQUALIZATION}_{MASK_TYPE}_{train folds}[_{EXP_TAG}]/`, for example
`runs/diffusion_avg_eq_paired_1234/`. `config.exp_name(fold)` builds it for any fold.

### 3.5 Train

Use `tmux` or `nohup`, because runs take many hours:

```bash
tmux new -s ct2mri
python train.py 2>&1 | tee runs/train_$(date +%F_%H%M).out
# detach: Ctrl-b d     reattach: tmux attach -t ct2mri
```

- Progress: one line per epoch on stdout, plus `runs/<exp>/train_log.csv`.
- Visual check: `runs/<exp>/samples/epoch_XXXX.png` (CT | rMRI | sMRI), written every `VAL_EVERY` epochs.
- **Resume:** run `python train.py` again. With `RESUME = True` it continues from
  `checkpoints/last.pt`, which is saved every `CKPT_EVERY` epochs.
- **Start over:** delete `runs/<exp>/` or set `RESUME = False`.
- Outputs: `checkpoints/{last,best,final}.pt`, `config.json`, `train_time.txt`.

### 3.6 Test

```bash
python test.py
```

This loads the EMA weights from `checkpoints/{CHECKPOINT_NAME}` (default `final.pt`), samples
every slice of `TEST_FOLD` with 50-step DDIM, and writes the following to `runs/<exp>/results/`:

- `results_<timestamp>.log`: the old format, `# fid, mse, mae, cs, psnr,ssim`. The SSIM column is
  **`ssim_legacy`**, so this file is directly comparable with the old logs.
- `metrics_summary_<timestamp>.csv`: mean and standard deviation of every metric (both SSIMs),
  FID, and sampling time.
- `per_image_<timestamp>.csv`: every metric for every slice.
- `predictions_fold<k>.npz`: `arr_0` CT, `arr_1` rMRI, `arr_2` sMRI, as `(N,256,256,1)` in
  [-1, 1], the same layout as the input data.
- `figures/<exp>_plot_<i>.png`: the first `NUM_TEST_FIGURES` slices.

The first run downloads the InceptionV3 weights (~100 MB) into the torch hub cache. If the
server has no internet access, download `inception_v3_google-0cc3c7bd.pth` elsewhere and
copy it to `~/.cache/torch/hub/checkpoints/`, or set `COMPUTE_FID = False`.

### 3.7 Full cross-validation

Each fold is a separate run. Edit `TEST_FOLD` and repeat, or loop:

```bash
for f in 1 2 3 4 5; do
  sed -i "s/^TEST_FOLD = .*/TEST_FOLD = $f/" config.py
  python train.py && python test.py
done
python aggregate.py      # mean ± std over the folds -> runs/diffusion_<dataset>[_<tag>]_cv_summary.csv
```

`aggregate.py` uses the newest `metrics_summary_*.csv` of each fold and skips folds without results.

### 3.8 Local smoke test (no GPU, no real data)

Before any long run after code changes, run the test suite from the repository root:

```bash
python -m pytest -q
```

It takes 2 to 4 minutes on a CPU and covers:

- `tests/test_scheduler.py`: the **oracle scheduler test**. A "model" that returns the exact target
  for a known x0 must reconstruct x0 for every combination of linear/cosine, eps/v and ddim/ddpm.
- `tests/test_metrics.py`: identical images give SSIM=1 and CS=1, known MSE/PSNR values, and the
  Fréchet distance matches analytic values.
- `tests/test_aggregate.py`: fold aggregation picks the newest summary per fold.
- `tests/test_smoke.py`: writes 5 synthetic folds to a temp dir, patches `config` (tiny UNet, CPU,
  `NUM_WORKERS=0`, `COMPUTE_FID=False`), runs `train.main()`, resumes it for one more epoch, then
  runs `test.py`, for both the paired and the masked dataset layout; plus a forward pass of the
  full-size default UNet at 256×256.

The smoke test loads `test.py` by file path, because `import test` can resolve to the stdlib package.

---

## 4. The earlier work: GAN-CM

The old repo (TensorFlow 2/Keras, `flags.py` + argparse) lives separately. Its relevant parts:

### 4.1 Data and preprocessing

- Paired axial CT/MRI slices, originally 512×512, downsampled to **256×256** for most experiments.
- Preprocessing happened in a notebook (`Resources/equalize.ipynb` on the server, not in the repo).
  Three intensity variants: `eq` (histogram equalization), `no_eq`, and `avg_eq`
  (averaged/adaptive equalization). Intensities were normalized to **[-1, 1]**
  (`norm_neg1pos1` / `norm_mask_neg1pos1` file names).
- Masks: `seg` is a binary segmentation computed from the original CT. `bin_edge` is a
  binarized Sobel edge map of the equalized CT (threshold 0.03). `paired` means no mask.
- Split into **5 cross-validation folds**, one `.npz` per fold: `arr_0` CT, `arr_1` MRI,
  `arr_2` mask (masked datasets only), all `(N,256,256,1)` float.
- Server location: `/media/aisec-102/DATA3/rachel/data/CV/{eq}_{mask}/`. The un-equalized originals
  are in `original_folds/`. A **separate external test set** exists at
  `/media/aisec-102/DATA3/rachel/data/test_data/{avg_eq,eq,no_eq}_seg_test.npz`
  (also referred to as `IMAGE-DataSet#1`).
- A Polaris (Argonne) copy lived at `/eagle/EVITA/ct-mri/data/CV/`.

### 4.2 Models

- **GAN-CM** (originally "PCxGAN"): a SPADE-style VAE-GAN.
  - *Encoder*: MRI to (mean, log-variance) of a 256-d latent; KL loss.
  - *Decoder/generator*: latent vector plus SPADE normalization conditioned on the
    mask (one-hot, 2 channels) and/or the CT.
  - *Discriminator*: multi-scale/patch discriminator conditioned on the CT; hinge loss.
  - *Generator losses*: VGG19 feature matching (×1), SSIM loss (×2), KL (×2.5 =
    50 × 0.05), plus discriminator feature matching in some variants.
  - Optimizers: generator Adam(2e-4, β=(0.0, 0.9)); discriminator Adam(2e-4, β=(0.5, 0.999)).
  - Variants: `gancm_just_ct`, `gancm_just_mask`, `gancm_mask_ct` (the main one),
    `gancm_512_mask_ct`.
  - **At inference the latent is sampled from N(0, 2²)**, not from the encoder, so outputs are stochastic.
- Baselines: **Pix2Pix**, **CycleGAN**, a plain **UNet**, **UViT** (a UNet-ViT trained with
  VGG + SSIM losses and a random "timestep" input; not a real diffusion model, since it had no
  noising/denoising process) and **UViT-SPADE**.

### 4.3 Training procedure

- Typically **500 epochs, batch size 1**. Training used folds ≠ `test_fold`, and the
  **test fold doubled as the validation set** (`validation_data=test_data`).
- Augmentation (the `DataGeneratorAug*` loaders): resize to 286 then random crop to 256,
  random h/v flips, brightness +0.3, 80% central crop, rot90, with the data repeated 3×.
  The old README reports that the resize/crop steps caused **pixelation**, and that removing all
  cropping and resizing gave the best results.
- Runs were launched per fold/GPU via small Python wrappers (`experiments/*.py`), or with PBS
  `qsub` on Polaris.

### 4.4 Evaluation procedure

- `evaluate.py`: for each test batch (batch size **1**), the output and target are rescaled to
  [0,1], then FID, MSE, MAE, CS, PSNR and SSIM are computed and **averaged over batches**, and saved
  with `np.savetxt` as `results_<epoch>_<timestamp>.log` with header `fid, mse, mae, cs, psnr,ssim`.
- `get_results.ipynb` (server, `Resources/`) averaged the five folds per experiment.
- `test_data_eval.py` evaluated the external test set. `get_predictions.py` saved images.
  DICOM conversion was done in notebooks (`process_512_data`, `save_test_predictions`,
  `convert_dicom_working`).

---

## 5. Metric definitions and quirks of the old code

This repo reproduces the old definitions where they are sound, and keeps legacy columns where
they are not.

| Metric | Old code | This repo |
| --- | --- | --- |
| MSE, MAE | sklearn on flattened [0,1] images | same, per image, then averaged |
| CS | sklearn cosine similarity of flattened images (returned as a 1×1 array) | `F.cosine_similarity` per image |
| PSNR | `20·log10(1/√MSE)` on [0,1] | same (`10·log10(1/MSE)`) |
| SSIM | `tf.image.ssim(max_val=1, 11, σ=1.5, k1=.01, k2=.03)` but **after a second `(x+1)/2` rescale**, so it was computed on [0.5,1] images, which **inflates the score** | `ssim`: correct, same Gaussian/valid-padding SSIM on [0,1]. `ssim_legacy`: reproduces the double rescale for comparison. |
| FID | Keras InceptionV3 **per batch of 1 image**, where the covariance is undefined; `preprocess_input` received [0,1] instead of [0,255] | One FID over the whole fold: torchvision ImageNet InceptionV3 pool features, resized to 299, ImageNet normalization. **Not comparable** with the old FID values. |

Averaging per-image values equals the old batch-size-1 averaging, so MSE, MAE, CS, PSNR and
`ssim_legacy` can be compared directly. Use `ssim` for anything new, and report both SSIM
values side by side when comparing with GAN-CM.

Other quirks of the old code worth knowing (so they aren't copied):

- `DataGenerator_Ready` / `DataGenerator_PairedReady` call `self.dataset.shuffle(...)` without
  assigning the result, so **the old training data was never shuffled**.
- The old augmentation applied brightness changes to the mask as well, and applied `rand_flip` to the
  mask through an `elif` chain that differs between loaders.
- Some scripts hard-code dataset folder names that don't match the README (`avg_eq_mask` vs `avg_eq_seg`).
- Hard-coded "bad image" index lists exist only for the 512 data. They aren't needed at 256.

---

## 6. Design of this project (and why)

- **Conditioning by concatenation**: the UNet input is `[x_t (MRI), CT, (mask)]`. The output is a
  single channel (ε or v). This is the standard image-to-image diffusion setup (as in SR3/Palette).
  The mask is binarized and mapped to {-1, 1}.
- **Defaults are the "standard DDPM"**: 1000 steps, linear β 1e-4 to 2e-2, ε-prediction, MSE.
  Alternatives available as config switches: cosine schedule, v-prediction, L1/Huber loss, Min-SNR-γ.
- **Sampling**: 50-step deterministic DDIM (`DDIM_ETA=0`) with x0 clipped to [-1,1], from the
  **EMA** weights (decay 0.9995). `SAMPLER="ddpm"` runs the full 1000-step ancestral sampler.
- **UNet**: base 64 channels, multipliers (1,1,2,2,4,4) over 256 down to 8, 2 ResBlocks per level,
  attention at 16 and 8 plus the bottleneck, GroupNorm(32), dropout 0.1, about **31M parameters**.
- **Optimization**: AdamW 1e-4, 1000-step linear warmup, then constant (or cosine), gradient
  clipping 1.0, bf16 autocast on CUDA, TF32 enabled.
- **Augmentation**: horizontal flip only (p=0.5) by default. vflip and rot90 are available but off,
  because they produce anatomically implausible orientations. Resize/crop jitter was deliberately
  dropped (it caused pixelation in GAN-CM).
- **Validation**: the denoising loss on the full test fold with fixed noise every epoch (cheap),
  plus sampled metrics on `VAL_NUM_SAMPLES` evenly spaced test-fold slices every `VAL_EVERY` epochs.
- **Checkpoint selection**: `best.pt` = lowest sampled validation MAE. Because it's chosen *on the
  test fold*, it's optimistically biased, so `test.py` evaluates `final.pt` by default (as the old
  code did, which evaluated the model at the end of training).
- **Reproducibility**: `SEED` fixes initialization, validation noise and the sampling noise in testing.

---

## 7. Status and remaining work

**Done and verified locally (CPU, synthetic data, `python -m pytest`: 17 passed):** all modules,
the oracle scheduler test, metric sanity checks, the full UNet forward pass at 256×256, fold
aggregation (`aggregate.py`), experiment tagging (`EXP_TAG`), and a complete train → resume →
test run for both paired and masked data that produced every output file. See `CHANGELOG.md`.

The legacy argparse prototype (`ct2mri-ddpm-main/` at the git root) was removed as a duplicate
of this codebase; it is still in git history (commit `070a8bd`) if anything needs to be looked up.

**Not yet done or verified:**

1. **First real run on the server.** Confirm the folder names, fold sizes and value ranges with
   `python data_loader.py`, and confirm the `cuda:1` mapping and memory use at `BATCH_SIZE=8`.
2. **FID path not exercised.** The InceptionV3 download and the `FIDCalculator` forward pass weren't
   run (only `frechet_distance` was tested).
3. **External test set evaluation**: support `/media/aisec-102/DATA3/rachel/data/test_data/*_seg_test.npz`
   (single file, not fold-based). Add a config constant such as `EXTERNAL_TEST_PATH` and a branch in `test.py`.
4. **Fair re-scoring of GAN-CM**: add a script that scores any predictions `.npz` (`arr_0` CT, `arr_1` rMRI,
   `arr_2` sMRI) with `metrics.py`. The old models' predictions can be exported *with the old repo, in its
   own environment*, and scored here, so both models get the corrected SSIM and the fold-level FID.
   (The new repo must still never import the old code.)
5. **Experiment grid** matching the old study: {eq, no_eq, avg_eq} × {paired, seg} (× bin_edge if needed) × 5 folds.
6. **Hyperparameter study**: v-prediction + cosine, Min-SNR γ=5, `SAMPLING_STEPS` (25/50/100/250),
   DDPM vs DDIM, EMA decay, number of epochs/steps, model width. Give each one its own `EXP_TAG`.
7. **Timing comparison**: report training time and *per-slice inference time* next to GAN-CM. Diffusion
   sampling is about `SAMPLING_STEPS` times slower than a single GAN forward pass.
8. Optional: TensorBoard/W&B logging, multi-sample inference (mean/variance maps across seeds for
   uncertainty), DICOM export from `predictions_fold<k>.npz`, a fp16 + GradScaler path for GPUs without bf16.

---

## 8. Known risks and likely problems

- **Folder names** may not match the config pattern exactly (old scripts use both `avg_eq_mask` and
  `avg_eq_seg`). Check with `ls` and adjust `MASK_TYPE`/`FILE_PREFIX` if needed.
- **Mask file contents**: the code assumes `arr_2` exists and is binary (thresholded at 0.5). If a dataset
  stores masks differently (one-hot, 0/255), fix `load_folds`.
- **Memory**: the whole training set is loaded into RAM (float32 tensors). A few thousand 256² slices
  are only a few hundred MB. `NUM_WORKERS` processes share it via fork on Linux, but on Windows each
  worker copies it.
- **GPU memory**: 31M parameters at batch 8 / 256² with bf16 is untested. If you hit out-of-memory errors,
  lower `BATCH_SIZE` (and possibly `LEARNING_RATE`), or reduce `BASE_CHANNELS`/`CHANNEL_MULT`.
- **bf16 needs Ampere or newer** (A100/RTX 30xx+). On V100/older GPUs set `USE_AMP = False`
  (fp16 + GradScaler isn't implemented).
- **GPU index drift**: if the server's GPUs are added, removed or reordered, `cuda:1` may point at a
  different card. Re-check with §3.2. Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes PyTorch use
  `nvidia-smi` order, but then **GPU 0 becomes `cuda:0`**, so change `DEVICE` too.
- **Shared server**: other users may occupy the GPU. Check `nvidia-smi` before launching.
- **Small dataset, big model**: overfitting is possible. Watch `val_loss` vs `train_loss` in `train_log.csv`
  and the sample figures. Dropout and flips are the only regularization.
- **Early validation looks bad**: EMA weights lag behind the live weights for the first several thousand
  steps, so sampled metrics in early epochs are not meaningful.
- **Validation on the test fold**: inherited from the old protocol. Anything chosen by looking at it
  (`best.pt`, hyperparameters) biases the test numbers. Keep `CHECKPOINT_NAME = "final.pt"` for reported results.
- **Stochastic outputs**: with `DDIM_ETA = 0` sampling is deterministic given the seed, but results still
  depend on `SEED`. For reported numbers, consider averaging over a few seeds.
- **`torch.load(weights_only=False)`** is used for checkpoints (they contain optimizer state and a config
  dict). Only load checkpoints you created.
- `test.py` shares its name with Python's stdlib `test` package. That's harmless unless something does
  `import test` expecting the stdlib. `pytest.ini` restricts pytest to `tests/`, and the smoke test loads
  `test.py` by path.
- Sampling the whole test fold costs `SAMPLING_STEPS` UNet passes per batch: budget for this in the CV loop.

---

## 9. Things to keep in mind

- Any change to preprocessing, normalization or metric code must keep results comparable with
  GAN-CM (§5). If a definition has to change, add a new column and keep the old one.
- Keep everything 256×256 and single-channel. Inputs and outputs are in [-1, 1]. Metrics rescale to [0, 1].
- The data on the server is the shared source of truth. **Never write to `DATA_ROOT`**: all outputs go to `runs/`.
- Large artifacts (`runs/`, checkpoints, `.npz` predictions) stay out of git.
- When comparing with GAN-CM, report: MSE, MAE, CS, PSNR, `ssim_legacy` (comparable), plus `ssim` and
  fold-level FID (new baseline), as mean ± std over the 5 folds, with training and inference time.

## 10. Code conventions

- Python 3.10+, PyTorch 2.x, flat module layout, run from the repo root.
- Settings come from `config` at call time (`config.X`), so scripts and tests can patch them.
- Tensors are NCHW float32. Data on disk is NHWC, and conversion happens only in `data_loader.py` and
  when `test.py` saves predictions.
- Match the existing style: short module docstrings explaining *what and why*, sparse inline comments,
  no dead or commented-out code, no duplicated variant files.
- Add new options as constants in `config.py` with a one-line comment listing the valid values.
