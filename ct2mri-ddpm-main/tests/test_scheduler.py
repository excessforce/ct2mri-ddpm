"""
Oracle test: a "model" that returns the exact target for a known x0 must let
every sampler reconstruct x0, for every schedule and prediction type.
"""

import pytest
import torch

from noise_scheduler import NoiseScheduler


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("prediction_type", ["eps", "v"])
@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_oracle_reconstructs_x0(schedule, prediction_type, sampler):
    scheduler = NoiseScheduler(1000, schedule, prediction_type=prediction_type)
    x0 = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(0)) * 1.6 - 0.8

    def oracle(inp, t):
        x_t = inp[:, :1]
        sqrt_ab = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = (x_t - sqrt_ab * x0) / sqrt_one_minus_ab
        return eps if prediction_type == "eps" else sqrt_ab * eps - sqrt_one_minus_ab * x0

    out = scheduler.sample(oracle, torch.zeros_like(x0), sampler, num_steps=50,
                           generator=torch.Generator().manual_seed(1))
    assert (out - x0).abs().max() < 1e-4


def test_v_target_round_trip():
    scheduler = NoiseScheduler(1000, "cosine", prediction_type="v")
    x0, noise = torch.rand(4, 1, 8, 8) * 2 - 1, torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 10, 500, 999])
    x_t = scheduler.add_noise(x0, noise, t)
    x0_hat, eps_hat = scheduler.predict_x0_and_eps(x_t, scheduler.get_target(x0, noise, t), t)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, noise, atol=1e-4)
