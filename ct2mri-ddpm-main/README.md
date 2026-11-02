# CT → MRI Diffusion

A PyTorch conditional denoising diffusion model that translates 256×256 CT slices
into MRI. It follows up on the GAN-CM work: it uses the same five-fold dataset and
reports the same metrics, so results can be compared directly. The code is
self-contained and does not import anything from the earlier project.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | Every setting: data path, folds, GPU, model size, schedule, training and testing options |
| `data_loader.py` | Loads the `.npz` folds into a PyTorch `Dataset` and `DataLoader`, with paired augmentation |
| `noise_scheduler.py` | Linear or cosine beta schedule, forward noising, eps/v targets, DDPM and DDIM samplers |
| `timestep_encoder.py` | Sinusoidal timestep embedding plus an MLP |
| `unet.py` | Conditional UNet: ResBlocks with timestep scale/shift, self-attention at 16×16 and 8×8 |
| `losses.py` | Diffusion loss (MSE, L1 or Huber), with optional Min-SNR weighting |
| `metrics.py` | MSE, MAE, cosine similarity, PSNR, SSIM, legacy SSIM, FID |
| `train.py` | Training loop with EMA, bf16 autocast, warmup, checkpoint/resume, periodic sampled validation |
| `test.py` | Generates an sMRI for every slice in the held-out fold and writes the metrics |
| `aggregate.py` | Averages the test results of the five folds (mean ± std across folds) |
| `utils.py` | Shared helpers: device, EMA, checkpoints, figures |
| `tests/` | pytest suite, including an end-to-end smoke test on synthetic data |
| `CHANGELOG.md` | Log of changes |

## Setup

```bash
pip install -r requirements.txt   # plus a CUDA build of torch for the server
```

### Data location

The data lives on the server. Set `DATA_ROOT` in `config.py`; the default is the
server path the earlier work used (`/media/aisec-102/DATA3/rachel/data/CV`).
`EQUALIZATION` and `MASK_TYPE` select the dataset folder (for example `avg_eq_paired`,
`eq_seg`), and the files are read from `{DATA_ROOT}/{EQUALIZATION}_{MASK_TYPE}/{FILE_PREFIX}{fold}.npz`.

If you run from another machine with the data mounted (for example via `sshfs`),
override the root without editing the file:

```bash
export CT2MRI_DATA_ROOT=/mnt/server/rachel/data/CV
```

To check the data before training, run this. It prints the slice count and value range of each fold:

```bash
python data_loader.py
```

### GPU

`DEVICE = "cuda:1"` in `config.py` (physical GPU 0 on the server shows up as `cuda:1` in PyTorch).
Do not also set `CUDA_VISIBLE_DEVICES`, because that renumbers the devices.

## Running

```bash
python -m pytest -q    # 2-4 min on CPU: smoke test on synthetic data, no GPU or real data needed
python train.py        # trains on every fold except TEST_FOLD; resumes from last.pt if one exists
python test.py         # evaluates CHECKPOINT_NAME (EMA weights) on TEST_FOLD
python aggregate.py    # after all five folds: mean ± std across folds
```

For the full five-fold cross-validation, change `TEST_FOLD` from 1 to 5 and run `train.py` and
`test.py` each time, then `aggregate.py` once. Each run writes to its own folder,
`runs/diffusion_{EQUALIZATION}_{MASK_TYPE}_{train folds}[_{EXP_TAG}]/`, which matches the earlier
naming (e.g. `uvit_avg_eq_1234`). Set `EXP_TAG` in `config.py` to give an ablation its own folder.

Outputs per run:

- `checkpoints/last.pt`, `best.pt`, `final.pt`
- `samples/epoch_XXXX.png`: CT | rMRI | sMRI figures from each validation run
- `train_log.csv`: train and validation loss per epoch, plus sampled validation metrics every `VAL_EVERY` epochs
- `results/results_<time>.log`: the earlier `fid, mse, mae, cs, psnr,ssim` format
- `results/metrics_summary_<time>.csv`, `results/per_image_<time>.csv`
- `results/predictions_fold<k>.npz`: `arr_0` CT, `arr_1` rMRI, `arr_2` sMRI, in the same NHWC layout as the input data

## Method

- **Conditioning**: the noisy MRI is concatenated channel-wise with the CT, plus the
  segmentation or edge mask when `MASK_TYPE` is not `paired`.
- **Training**: 1000-step linear schedule with ε-prediction and an MSE loss (standard DDPM).
  `PREDICTION_TYPE = "v"`, `BETA_SCHEDULE = "cosine"` and `MIN_SNR_GAMMA = 5.0` are the usual alternatives.
- **Sampling**: 50-step deterministic DDIM from the EMA weights. Set `SAMPLER = "ddpm"` to use all
  1000 ancestral steps, which is slower and stochastic.
- **Augmentation**: horizontal flips only by default. The earlier resize and crop jitter was left
  out because its README notes that it caused pixelation.

## Comparing with the earlier results

The metric definitions match the earlier `evaluate.py`. Images are rescaled from [-1, 1] to [0, 1],
each metric is computed per slice, and the values are averaged over the fold, which is the same as
the earlier batch-size-1 evaluation. PSNR uses a peak value of 1. SSIM uses an 11×11 Gaussian window
with σ = 1.5, k1 = 0.01, k2 = 0.03, the same parameters as `tf.image.ssim`.

Two differences from the earlier code need care:

1. **SSIM.** The earlier `get_metrics` received images that had already been rescaled to [0, 1] and
   applied `(x + 1) / 2` again before SSIM. That compresses both images into [0.5, 1] and inflates
   the score. This project reports both values:
   - `ssim`: the correct value, to use in new work.
   - `ssim_legacy`: the same computation as the earlier code, to compare against previously
     reported numbers. The `results_*.log` file uses `ssim_legacy` so it can be compared directly
     with the earlier logs.
2. **FID.** The earlier FID was computed per batch with a batch size of 1. A covariance matrix
   cannot be estimated from one sample, and the Keras `preprocess_input` received inputs in [0, 1]
   rather than the [0, 255] it expects. This project computes one FID over the whole fold using
   torchvision's ImageNet InceptionV3 (2048-d pool features, 299×299 input). The weights download
   on first use, so if the server has no internet access, copy them into the torch hub cache first.
   Treat FID as a new baseline rather than a comparison with the earlier numbers.

Diffusion sampling is stochastic, so `SEED` fixes the sampling noise for reproducible test results.

`best.pt` is chosen by MAE on a small subset of the test fold, which is optimistic because the test
fold is used for selection. `test.py` therefore evaluates `final.pt` by default.
