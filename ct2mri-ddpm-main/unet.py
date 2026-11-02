"""
Conditional UNet for image-to-image diffusion.

The network input is the noisy MRI concatenated channel-wise with the
conditioning images (CT, and the mask if enabled). Each residual block is
modulated by the timestep embedding (scale/shift after GroupNorm), and
multi-head self-attention is applied at the lower resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from timestep_encoder import TimestepEncoder


def norm(channels, groups):
    return nn.GroupNorm(min(groups, channels), channels)


def zero_init(module):
    """Zero the last layer of a residual branch so each block starts as identity."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout, groups):
        super().__init__()
        self.norm1 = norm(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = norm(out_ch, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, groups):
        super().__init__()
        self.num_heads = num_heads
        self.norm = norm(channels, groups)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_init(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        # (B, heads, head_dim, HW) -> (B, heads, HW, head_dim)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Level(nn.Module):
    """A residual block optionally followed by self-attention."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, groups, attention, num_heads):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout, groups)
        self.attn = AttentionBlock(out_ch, num_heads, groups) if attention else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=256, base_channels=64,
                 channel_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attention_resolutions=(16, 8),
                 num_heads=4, dropout=0.1, groups=32):
        super().__init__()
        emb_dim = base_channels * 4
        self.time_encoder = TimestepEncoder(base_channels, emb_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels = [base_channels]
        ch, res = base_channels, image_size
        for i, mult in enumerate(channel_mult):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(Level(ch, base_channels * mult, emb_dim, dropout, groups,
                                    res in attention_resolutions, num_heads))
                ch = base_channels * mult
                skip_channels.append(ch)
            self.down_blocks.append(blocks)
            if i < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                skip_channels.append(ch)
                res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck
        self.mid_res1 = ResBlock(ch, ch, emb_dim, dropout, groups)
        self.mid_attn = AttentionBlock(ch, num_heads, groups)
        self.mid_res2 = ResBlock(ch, ch, emb_dim, dropout, groups)

        # Decoder: one extra block per level consumes the downsampler's skip.
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(Level(ch + skip_channels.pop(), base_channels * mult, emb_dim, dropout,
                                    groups, res in attention_resolutions, num_heads))
                ch = base_channels * mult
            self.up_blocks.append(blocks)
            if i > 0:
                self.upsamplers.append(Upsample(ch))
                res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        self.out = nn.Sequential(
            norm(ch, groups),
            nn.SiLU(),
            zero_init(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t):
        emb = self.time_encoder(t)

        h = self.in_conv(x)
        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, emb)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        for blocks, up in zip(self.up_blocks, self.upsamplers):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), emb)
            h = up(h)

        return self.out(h)


def build_unet():
    return UNet(
        in_channels=config.IMG_CHANNELS + config.COND_CHANNELS,
        out_channels=config.IMG_CHANNELS,
        image_size=config.IMAGE_SIZE,
        base_channels=config.BASE_CHANNELS,
        channel_mult=config.CHANNEL_MULT,
        num_res_blocks=config.NUM_RES_BLOCKS,
        attention_resolutions=config.ATTENTION_RESOLUTIONS,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        groups=config.GROUP_NORM_GROUPS,
    )
