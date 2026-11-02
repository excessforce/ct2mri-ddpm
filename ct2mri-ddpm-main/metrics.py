"""
Evaluation metrics, matching the definitions used in the prior GAN work.

All per-image metrics take tensors in the model's [-1, 1] range and rescale to
[0, 1] first, as the prior evaluation did. Per-image values are averaged over
the test fold, which equals the prior batch-size-1 evaluation.

    mse, mae      over all pixels of an image
    cs            cosine similarity of the flattened images
    psnr          20 * log10(1 / sqrt(mse)), max value 1
    ssim          Gaussian SSIM (11x11, sigma 1.5, k1 0.01, k2 0.03, valid padding),
                  same parameters as tf.image.ssim
    ssim_legacy   the prior code rescaled images to [0, 1] and then applied
                  (x + 1) / 2 a second time before SSIM, i.e. computed SSIM on
                  [0.5, 1] images with max_val 1. Reproduced here only so the
                  numbers can be compared with previously reported results.
    fid           Frechet Inception Distance over the whole fold (see FIDCalculator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_unit_range(x):
    return ((x.float() + 1.0) / 2.0).clamp(0, 1)


def _gaussian_kernel(size=11, sigma=1.5, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, max_val=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """Per-image SSIM for (B, 1, H, W) tensors already in [0, max_val]."""
    kernel = _gaussian_kernel(size, sigma, x.device)
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2

    mu_x = F.conv2d(x, kernel)
    mu_y = F.conv2d(y, kernel)
    sigma_x = F.conv2d(x * x, kernel) - mu_x ** 2
    sigma_y = F.conv2d(y * y, kernel) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, kernel) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def image_metrics(real, fake):
    """
    Per-image metrics for (B, 1, H, W) tensors in [-1, 1].
    Returns a dict of (B,) tensors.
    """
    real, fake = to_unit_range(real), to_unit_range(fake)
    r, f = real.flatten(1), fake.flatten(1)

    mse = ((r - f) ** 2).mean(dim=1)
    mae = (r - f).abs().mean(dim=1)
    cs = F.cosine_similarity(r, f, dim=1, eps=1e-8)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-12))

    return {
        "mse": mse,
        "mae": mae,
        "cs": cs,
        "psnr": psnr,
        "ssim": ssim(real, fake),
        "ssim_legacy": ssim((real + 1) / 2, (fake + 1) / 2),
    }


class FIDCalculator:
    """
    FID between the real and generated MRI distributions of a whole fold.

    Uses torchvision's ImageNet InceptionV3 pool features (2048-d). Grayscale
    images are repeated to three channels, resized to 299x299 and normalized
    with ImageNet statistics. The prior work computed FID per batch with a
    batch size of 1, where the covariance is undefined, so its FID values are
    not directly comparable with this fold-level FID.
    """

    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        self.model = model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.real_feats, self.fake_feats = [], []

    @torch.no_grad()
    def _features(self, x):
        x = to_unit_range(x).to(self.device).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model((x - self.mean) / self.std).double().cpu()

    def update(self, real, fake):
        self.real_feats.append(self._features(real))
        self.fake_feats.append(self._features(fake))

    def compute(self):
        return frechet_distance(torch.cat(self.real_feats), torch.cat(self.fake_feats))


def _sqrtm_psd(mat):
    vals, vecs = torch.linalg.eigh(mat)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T


def frechet_distance(feats1, feats2):
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), using float64 features."""
    mu1, mu2 = feats1.mean(0), feats2.mean(0)
    s1, s2 = torch.cov(feats1.T), torch.cov(feats2.T)
    # Tr((S1 S2)^(1/2)) = Tr((sqrt(S1) S2 sqrt(S1))^(1/2)), which is symmetric PSD.
    sqrt_s1 = _sqrtm_psd(s1)
    tr_covmean = torch.linalg.eigvalsh(sqrt_s1 @ s2 @ sqrt_s1).clamp_min(0).sqrt().sum()
    return float(((mu1 - mu2) ** 2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * tr_covmean)
