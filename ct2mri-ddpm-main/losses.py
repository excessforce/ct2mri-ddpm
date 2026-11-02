"""
Diffusion training loss.

The network predicts either the added noise (eps) or the velocity (v); the
loss compares that prediction with the scheduler's target. Optional Min-SNR
weighting (Hang et al., 2023) down-weights the easy low-noise timesteps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, scheduler, loss_type="mse", min_snr_gamma=None):
        super().__init__()
        if loss_type not in ("mse", "l1", "huber"):
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.scheduler = scheduler
        self.loss_type = loss_type
        self.min_snr_gamma = min_snr_gamma

    def _elementwise(self, pred, target):
        if self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none")
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        return F.smooth_l1_loss(pred, target, reduction="none", beta=0.1)

    def _snr_weights(self, t):
        snr = self.scheduler.snr.to(t.device)[t]
        # The v-target already carries a (snr + 1) factor relative to eps.
        denom = snr + 1 if self.scheduler.prediction_type == "v" else snr
        return snr.clamp(max=self.min_snr_gamma) / denom

    def forward(self, pred, target, t):
        per_sample = self._elementwise(pred.float(), target.float()).mean(dim=(1, 2, 3))
        if self.min_snr_gamma is not None:
            per_sample = per_sample * self._snr_weights(t)
        return per_sample.mean()
