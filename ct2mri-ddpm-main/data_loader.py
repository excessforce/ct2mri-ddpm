"""
Dataset and DataLoader for the paired CT/MRI cross-validation folds.

Each fold is a .npz file at f"{config.DATA_PATH_PREFIX}{fold}.npz" with
    arr_0: CT   (N, 256, 256, 1), normalized to [-1, 1]
    arr_1: MRI  (N, 256, 256, 1), normalized to [-1, 1]
    arr_2: mask (N, 256, 256, 1), binary, only in the "seg"/"bin_edge" datasets

Arrays are converted to channels-first float32 tensors: (N, 1, 256, 256).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config


def fold_path(fold):
    return f"{config.DATA_PATH_PREFIX}{fold}.npz"


def _to_nchw(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:                       # (N, H, W)
        arr = arr[:, None]
    elif arr.ndim == 4 and arr.shape[-1] in (1, 2, 3):
        arr = arr.transpose(0, 3, 1, 2)     # NHWC -> NCHW
    return torch.from_numpy(np.ascontiguousarray(arr))


def load_folds(folds, use_mask=None):
    """Load and concatenate the requested folds into CT, MRI and (optional) mask tensors."""
    if use_mask is None:
        use_mask = config.USE_MASK
    cts, mris, masks = [], [], []
    for fold in folds:
        with np.load(fold_path(fold)) as data:
            cts.append(_to_nchw(data["arr_0"]))
            mris.append(_to_nchw(data["arr_1"]))
            if use_mask:
                if "arr_2" not in data:
                    raise KeyError(f"USE_MASK is set but {fold_path(fold)} has no arr_2")
                masks.append(_to_nchw(data["arr_2"]))

    ct, mri = torch.cat(cts), torch.cat(mris)
    # Binarize and map to [-1, 1] so the mask shares the image value range.
    mask = (torch.cat(masks) > 0.5).float() * 2 - 1 if use_mask else None

    expected = (1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    if ct.shape[1:] != expected or mri.shape[1:] != expected:
        raise ValueError(f"Expected images of shape {expected}, got CT {tuple(ct.shape[1:])} "
                         f"and MRI {tuple(mri.shape[1:])}")
    return ct, mri, mask


class CTMRIDataset(Dataset):
    """Paired CT/MRI slices. Returns a dict with "ct", "mri" and optionally "mask"."""

    def __init__(self, folds, augment=False):
        self.ct, self.mri, self.mask = load_folds(folds)
        self.augment = augment

    def __len__(self):
        return self.ct.shape[0]

    def _augment(self, images):
        # Same random transform for every image of the pair.
        if torch.rand(()) < config.AUG_HFLIP_P:
            images = [im.flip(-1) for im in images]
        if torch.rand(()) < config.AUG_VFLIP_P:
            images = [im.flip(-2) for im in images]
        if torch.rand(()) < config.AUG_ROT90_P:
            k = int(torch.randint(1, 4, ()))
            images = [im.rot90(k, dims=(-2, -1)) for im in images]
        return images

    def __getitem__(self, idx):
        images = [self.ct[idx], self.mri[idx]]
        if self.mask is not None:
            images.append(self.mask[idx])
        if self.augment:
            images = self._augment(images)

        sample = {"ct": images[0], "mri": images[1], "index": idx}
        if self.mask is not None:
            sample["mask"] = images[2]
        return sample


def build_condition(batch):
    """Stack the conditioning channels (CT, plus mask if used) along the channel dim."""
    if "mask" in batch:
        return torch.cat([batch["ct"], batch["mask"]], dim=1)
    return batch["ct"]


def get_dataloaders():
    """Training loader over TRAIN_FOLDS and an un-augmented loader over TEST_FOLD."""
    train_set = CTMRIDataset(config.TRAIN_FOLDS, augment=config.AUGMENT)
    test_set = CTMRIDataset([config.TEST_FOLD], augment=False)
    return get_train_loader(train_set), get_test_loader(test_set)


def get_train_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )


def get_test_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Sanity check: print shapes and value ranges of every fold.
    for fold in range(1, config.NUM_FOLDS + 1):
        ct, mri, mask = load_folds([fold])
        line = (f"fold {fold}: {ct.shape[0]} slices | "
                f"CT [{ct.min():.3f}, {ct.max():.3f}] | MRI [{mri.min():.3f}, {mri.max():.3f}]")
        if mask is not None:
            line += f" | mask foreground {(mask > 0).float().mean():.3f}"
        print(line)
