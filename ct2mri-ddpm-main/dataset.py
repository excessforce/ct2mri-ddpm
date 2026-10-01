"""PyTorch data pipeline for paired CT -> MRI slices (replaces the old TF data_loader.py).

Data layout (same as the old repo): .npz files with arr_0 = CT, arr_1 = MRI,
optional arr_2 = mask; arrays shaped (N, H, W, 1) or (N, H, W).
Everything is converted to float32 tensors in [-1, 1], shape (N, 1, H, W).
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Indices into the RAW array of the single-file dataset (flags.data_path in the old
# code), before any shuffling/splitting. Never applied to the 5-fold files.
BAD_IMAGE_IDX = (37, 42, 111, 907, 908, 936, 1108, 1110, 1116, 1117, 1130, 1138,
                 1545, 1557, 2006, 2012, 2021, 1925, 1926, 1932, 1938, 950)


# loading
def _to_nchw(a, name, input_range):
    """(N,H,W) or (N,H,W,1) numpy -> (N,1,H,W) float tensor scaled to [-1, 1]."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[:, None]
    elif a.ndim == 4 and a.shape[-1] == 1:
        a = a.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"{name}: unexpected shape {a.shape}")
    lo, hi = input_range
    if a.min() < lo - 1e-3 or a.max() > hi + 1e-3:
        raise ValueError(f"{name}: values span [{a.min():.3f}, {a.max():.3f}] "
                         f"but input_range={input_range}; set input_range correctly.")
    return torch.from_numpy((a - lo) / (hi - lo) * 2 - 1)


def load_npz(paths, with_mask=False):
    """Load and concatenate one or more .npz files -> (ct, mri, mask|None) numpy arrays."""
    parts = [np.load(p) for p in paths]
    ct = np.concatenate([d["arr_0"] for d in parts])
    mri = np.concatenate([d["arr_1"] for d in parts])
    mask = np.concatenate([d["arr_2"] for d in parts]) if with_mask else None
    return ct, mri, mask


def drop_bad_images(arrays, bad_idx=BAD_IMAGE_IDX):
    """Delete bad indices from every array. Fails loudly instead of silently skipping."""
    n = len(arrays[0])
    if max(bad_idx) >= n:
        raise ValueError(f"bad-image list has index {max(bad_idx)} but the array has only "
                         f"{n} images: the list belongs to a different file.")
    return [None if a is None else np.delete(a, list(bad_idx), axis=0) for a in arrays]


# dataset
class PairedSliceDataset(Dataset):
    """Paired CT/MRI slices with optional mask. Augmentations are applied jointly."""

    def __init__(self, ct, mri, mask=None, augment=False, input_range=(0.0, 1.0),
                 image_size=None, p_flip_h=0.35, p_flip_v=0.35, p_rot90=0.5,
                 p_zoom=0.0, min_scale=0.8):  # zoom off: resizing caused pixelation in the GAN
        self.ct = _to_nchw(ct, "ct", input_range)
        self.mri = _to_nchw(mri, "mri", input_range)
        self.mask = None
        if mask is not None:
            self.mask = (_to_nchw(mask, "mask", (0.0, 1.0)) > 0).float()  # binary {0,1}
        assert len(self.ct) == len(self.mri), "CT/MRI count mismatch"

        if image_size is not None and self.ct.shape[-1] != image_size:
            size = (image_size, image_size)
            kw = dict(size=size, mode="bilinear", antialias=True, align_corners=False)
            self.ct, self.mri = F.interpolate(self.ct, **kw), F.interpolate(self.mri, **kw)
            if self.mask is not None:
                self.mask = F.interpolate(self.mask, size=size, mode="nearest")

        self.augment = augment
        self.p_flip_h, self.p_flip_v, self.p_rot90 = p_flip_h, p_flip_v, p_rot90
        self.p_zoom, self.min_scale = p_zoom, min_scale

    def __len__(self):
        return len(self.ct)

    def _resize(self, t, size):
        img = F.interpolate(t[None, :2], size=size, mode="bilinear", align_corners=False)[0]
        if t.shape[0] == 2:
            return img
        m = F.interpolate(t[None, 2:], size=size, mode="nearest")[0]
        return torch.cat([img, m], 0)

    def _augment(self, t):  # t: (C, H, W), channels = [ct, mri, (mask)]
        H, W = t.shape[-2:]
        if random.random() < self.p_flip_h:
            t = t.flip(-1)
        if random.random() < self.p_flip_v:
            t = t.flip(-2)
        if H == W and random.random() < self.p_rot90:
            t = torch.rot90(t, 1, dims=(-2, -1))
        if random.random() < self.p_zoom:  # random crop of 80-100% then resize back
            s = random.uniform(self.min_scale, 1.0)
            h, w = int(H * s), int(W * s)
            top, left = random.randint(0, H - h), random.randint(0, W - w)
            t = self._resize(t[..., top:top + h, left:left + w], (H, W))
        return t

    def __getitem__(self, i):
        chans = [self.ct[i], self.mri[i]] + ([self.mask[i]] if self.mask is not None else [])
        t = torch.cat(chans, 0)
        if self.augment:
            t = self._augment(t)
        out = {"ct": t[0:1], "mri": t[1:2]}
        if self.mask is not None:
            out["mask"] = t[2:3]
        return out


# builders
def build_fold_datasets(prefix, test_fold, val_fold=None, with_mask=False, folds=range(1, 6),
                        **ds_kwargs):
    """5-fold layout: files f'{prefix}{i}.npz'. Returns (train, val|None, test)."""
    held_out = {test_fold, val_fold}
    train_folds = [f for f in folds if f not in held_out]

    def make(fold_ids, augment):
        ct, mri, mask = load_npz([f"{prefix}{i}.npz" for i in fold_ids], with_mask)
        return PairedSliceDataset(ct, mri, mask, augment=augment, **ds_kwargs)

    val = make([val_fold], False) if val_fold is not None else None
    return make(train_folds, True), val, make([test_fold], False)


def build_single_file_datasets(path, with_mask=False, remove_bad=True, val_rate=0.1,
                               test_rate=0.1, seed=42, **ds_kwargs):
    """One big .npz, split by a seeded permutation (the old split was unseeded)."""
    ct, mri, mask = load_npz([path], with_mask)
    if remove_bad:
        ct, mri, mask = drop_bad_images([ct, mri, mask])
    perm = np.random.RandomState(seed).permutation(len(ct))
    n_test, n_val = int(len(ct) * test_rate), int(len(ct) * val_rate)
    parts = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}

    def make(idx, augment):
        return PairedSliceDataset(ct[idx], mri[idx], None if mask is None else mask[idx],
                                  augment=augment, **ds_kwargs)

    return make(parts["train"], True), make(parts["val"], False), make(parts["test"], False)


def make_loader(ds, batch_size, train, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)