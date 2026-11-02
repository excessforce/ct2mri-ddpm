"""
Gaussian diffusion noise scheduler.

Implements the forward process q(x_t | x_0), the training targets for
epsilon- and v-prediction, and two reverse-process samplers:
    - DDPM (Ho et al., 2020): ancestral sampling over every training step
    - DDIM (Song et al., 2021): strided sampling, deterministic when eta = 0
"""

import math

import numpy as np
import torch
from tqdm import tqdm


def make_betas(schedule, num_timesteps, beta_start=1e-4, beta_end=2e-2):
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    if schedule == "cosine":
        # Nichol & Dhariwal (2021), with betas clipped to avoid a singular final step.
        s = 0.008
        steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
        alphas_cumprod = np.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, 0, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


def _extract(arr, t, ndim):
    """Gather per-timestep coefficients and reshape them to broadcast over images."""
    out = arr.to(t.device)[t]
    return out.view(-1, *([1] * (ndim - 1)))


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_schedule="linear", beta_start=1e-4,
                 beta_end=2e-2, prediction_type="eps"):
        if prediction_type not in ("eps", "v"):
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        betas = make_betas(beta_schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Coefficients are computed in float64 and stored as float32 tensors.
        def as_tensor(x):
            return torch.tensor(x, dtype=torch.float32)

        self.betas = as_tensor(betas)
        self.alphas_cumprod = as_tensor(alphas_cumprod)
        self.sqrt_alphas_cumprod = as_tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = as_tensor(np.sqrt(1.0 - alphas_cumprod))
        self.snr = as_tensor(alphas_cumprod / (1.0 - alphas_cumprod))

        # Posterior q(x_{t-1} | x_t, x_0) used by the DDPM sampler.
        self.posterior_variance = as_tensor(
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef1 = as_tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = as_tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Forward process and training targets
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size, device, generator=None):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, generator=generator)

    def add_noise(self, x0, noise, t):
        """x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * noise"""
        return (_extract(self.sqrt_alphas_cumprod, t, x0.ndim) * x0
                + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * noise)

    def get_target(self, x0, noise, t):
        if self.prediction_type == "eps":
            return noise
        # v = sqrt(a_bar_t) * noise - sqrt(1 - a_bar_t) * x0
        return (_extract(self.sqrt_alphas_cumprod, t, x0.ndim) * noise
                - _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * x0)

    def predict_x0_and_eps(self, x_t, model_out, t):
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t.ndim)
        sqrt_one_minus_ab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        if self.prediction_type == "eps":
            eps = model_out
            x0 = (x_t - sqrt_one_minus_ab * eps) / sqrt_ab
        else:
            x0 = sqrt_ab * x_t - sqrt_one_minus_ab * model_out
            eps = sqrt_one_minus_ab * x_t + sqrt_ab * model_out
        return x0, eps

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def ddpm_step(self, x_t, model_out, t, generator=None, clip=True):
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x0, _ = self.predict_x0_and_eps(x_t, model_out, t_batch)
        if clip:
            x0 = x0.clamp(-1, 1)
        mean = (_extract(self.posterior_mean_coef1, t_batch, x_t.ndim) * x0
                + _extract(self.posterior_mean_coef2, t_batch, x_t.ndim) * x_t)
        if t == 0:
            return mean
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)
        return mean + _extract(self.posterior_variance, t_batch, x_t.ndim).sqrt() * noise

    def ddim_step(self, x_t, model_out, t, t_prev, eta=0.0, generator=None, clip=True):
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x0, eps = self.predict_x0_and_eps(x_t, model_out, t_batch)
        if clip:
            x0 = x0.clamp(-1, 1)
            # Re-derive epsilon from the clipped x0 so the update stays consistent.
            eps = ((x_t - _extract(self.sqrt_alphas_cumprod, t_batch, x_t.ndim) * x0)
                   / _extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x_t.ndim))

        ab_t = self.alphas_cumprod[t].item()
        ab_prev = self.alphas_cumprod[t_prev].item() if t_prev >= 0 else 1.0
        sigma = eta * math.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))

        x_prev = math.sqrt(ab_prev) * x0 + math.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
        if sigma > 0:
            x_prev = x_prev + sigma * torch.randn(x_t.shape, generator=generator, device=x_t.device)
        return x_prev

    def sampling_timesteps(self, sampler, num_steps):
        if sampler == "ddpm":
            return list(range(self.num_timesteps - 1, -1, -1))
        steps = np.linspace(0, self.num_timesteps - 1, num_steps).round().astype(int)
        return sorted(set(steps.tolist()), reverse=True)

    @torch.no_grad()
    def sample(self, model, cond, sampler="ddim", num_steps=50, eta=0.0, generator=None,
               out_channels=1, progress=False):
        """
        Generate images conditioned on `cond` (B, C_cond, H, W).
        `model(x, t)` receives the noisy image concatenated with the condition.
        """
        b, _, h, w = cond.shape
        x = torch.randn((b, out_channels, h, w), generator=generator, device=cond.device)
        timesteps = self.sampling_timesteps(sampler, num_steps)

        for i, t in enumerate(tqdm(timesteps, disable=not progress, leave=False, desc="sampling")):
            t_batch = torch.full((b,), t, device=cond.device, dtype=torch.long)
            model_out = model(torch.cat([x, cond], dim=1), t_batch).float()
            if sampler == "ddpm":
                x = self.ddpm_step(x, model_out, t, generator)
            else:
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                x = self.ddim_step(x, model_out, t, t_prev, eta, generator)
        return x.clamp(-1, 1)
