"""Sanity checks for the evaluation metrics."""

import torch

from metrics import frechet_distance, image_metrics


def test_identical_images_are_perfect():
    x = torch.rand(3, 1, 64, 64) * 2 - 1
    m = image_metrics(x, x)
    assert torch.allclose(m["ssim"], torch.ones(3), atol=1e-5)
    assert torch.allclose(m["ssim_legacy"], torch.ones(3), atol=1e-5)
    assert torch.allclose(m["cs"], torch.ones(3), atol=1e-5)
    assert torch.all(m["mse"] == 0) and torch.all(m["mae"] == 0)


def test_known_mse_and_psnr():
    real = torch.full((1, 1, 32, 32), -1.0)    # 0 in [0, 1]
    fake = torch.full((1, 1, 32, 32), 0.0)     # 0.5 in [0, 1]
    m = image_metrics(real, fake)
    assert torch.isclose(m["mse"], torch.tensor([0.25])).all()
    assert torch.isclose(m["mae"], torch.tensor([0.5])).all()
    assert torch.isclose(m["psnr"], torch.tensor([10 * torch.log10(torch.tensor(4.0))])).all()


def test_frechet_distance_matches_analytic_values():
    feats = torch.randn(500, 16, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert abs(frechet_distance(feats, feats)) < 1e-6
    # Equal covariances, so the distance is just the squared mean shift.
    shift = torch.full((16,), 0.5, dtype=torch.float64)
    assert abs(frechet_distance(feats, feats + shift) - float((shift ** 2).sum())) < 1e-6
