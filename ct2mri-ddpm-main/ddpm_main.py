"""Conditional DDPM for CT -> MRI synthesis (2D slices, PyTorch).

The CT slice is concatenated with the noisy MRI along the channel axis, so the
UNet learns to denoise the MRI while "looking at" the CT (Palette-style).
Images are expected in [-1, 1], shape (B, 1, H, W), H and W divisible by 8.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# utils
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def gn(ch, groups=8):
    return nn.GroupNorm(groups, ch)


# modules
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1, self.conv1 = gn(in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2, self.conv2 = gn(out_ch), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm, self.heads = gn(ch), heads
        self.qkv, self.out = nn.Conv2d(ch, ch * 3, 1), nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(o.transpose(-1, -2).reshape(B, C, H, W))


class Block(nn.Module):
    """ResBlock optionally followed by self-attention."""
    def __init__(self, in_ch, out_ch, t_dim, attn=False, dropout=0.1):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch) if attn else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# UNet
class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=64, mults=(1, 2, 4, 8),
                 attn_levels=(2, 3), n_res=2, dropout=0.1):
        super().__init__()
        self.base, t_dim = base, base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down, chs, ch = nn.ModuleList(), [base], base
        for lvl, m in enumerate(mults):
            for _ in range(n_res):
                self.down.append(Block(ch, base * m, t_dim, lvl in attn_levels, dropout))
                ch = base * m
                chs.append(ch)
            if lvl < len(mults) - 1:
                self.down.append(Downsample(ch))
                chs.append(ch)

        self.mid1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, t_dim, dropout)

        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mults))):
            for i in range(n_res + 1):
                self.up.append(Block(ch + chs.pop(), base * m, t_dim, lvl in attn_levels, dropout))
                ch = base * m
                if lvl > 0 and i == n_res:
                    self.up.append(Upsample(ch))

        self.out = nn.Sequential(gn(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))

    def forward(self, x_t, cond, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        hs = [h]
        for layer in self.down:
            h = layer(h) if isinstance(layer, Downsample) else layer(h, t_emb)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)
        for layer in self.up:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                h = layer(torch.cat([h, hs.pop()], dim=1), t_emb)
        return self.out(h)


# diffusion
class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.model, self.T = model, timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
        reg = lambda n, v: self.register_buffer(n, v)
        reg("betas", betas)
        reg("sqrt_ac", ac.sqrt())
        reg("sqrt_1m_ac", (1 - ac).sqrt())
        reg("sqrt_recip_ac", (1 / ac).sqrt())
        reg("sqrt_recipm1_ac", (1 / ac - 1).sqrt())
        reg("post_var", betas * (1 - ac_prev) / (1 - ac))
        reg("post_c1", betas * ac_prev.sqrt() / (1 - ac))
        reg("post_c2", (1 - ac_prev) * alphas.sqrt() / (1 - ac))

    @staticmethod
    def _at(buf, t):
        return buf[t][:, None, None, None]

    def loss(self, mri, ct):
        t = torch.randint(0, self.T, (mri.size(0),), device=mri.device)
        noise = torch.randn_like(mri)
        x_t = self._at(self.sqrt_ac, t) * mri + self._at(self.sqrt_1m_ac, t) * noise
        return F.mse_loss(self.model(x_t, ct, t), noise)

    @torch.no_grad()
    def sample(self, ct):
        x = torch.randn_like(ct)
        for i in reversed(range(self.T)):
            t = torch.full((ct.size(0),), i, device=ct.device, dtype=torch.long)
            eps = self.model(x, ct, t)
            x0 = (self._at(self.sqrt_recip_ac, t) * x - self._at(self.sqrt_recipm1_ac, t) * eps).clamp(-1, 1)
            mean = self._at(self.post_c1, t) * x0 + self._at(self.post_c2, t) * x
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + self._at(self.post_var, t).sqrt() * noise
        return x

    @torch.no_grad()
    def ddim_sample(self, ct, steps=50):
        """Deterministic DDIM sampling on a strided schedule (far faster than 1000 steps)."""
        ts = torch.linspace(self.T - 1, 0, steps).long().tolist()
        x = torch.randn_like(ct)
        for j, i in enumerate(ts):
            t = torch.full((ct.size(0),), i, device=ct.device, dtype=torch.long)
            eps = self.model(x, ct, t)
            x0 = (self._at(self.sqrt_recip_ac, t) * x - self._at(self.sqrt_recipm1_ac, t) * eps).clamp(-1, 1)
            if j == len(ts) - 1:
                return x0
            eps = (self._at(self.sqrt_recip_ac, t) * x - x0) / self._at(self.sqrt_recipm1_ac, t)
            nxt = ts[j + 1]
            x = self.sqrt_ac[nxt] * x0 + self.sqrt_1m_ac[nxt] * eps
        return x


if __name__ == "__main__":
    unet = UNet()
    ddpm = GaussianDiffusion(unet, timesteps=1000)
    ct, mri = torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128)
    print("loss:", ddpm.loss(mri, ct).item())  # one training step = loss.backward(); opt.step()