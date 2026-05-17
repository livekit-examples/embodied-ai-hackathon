"""Sparse-reward dataset on top of a LeRobotDataset.

Each frame is labelled 1 if it lies in the final `terminal_window` frames
of its episode (default 15 frames ≈ 0.5 s at 30 fps), else 0. The model
sees the same single-step inputs as ACT — one camera frame plus the 7-d
proprio state — with no temporal buffer.

Augmentations:
  * Image: random resized crop, color jitter, gaussian blur, random
    erasing, plus tiny affine jitter. No horizontal flip — flipping would
    invert gripper geometry and confuse the model.
  * State: small Gaussian noise added to proprio.
  * Label smoothing: the terminal boundary is fuzzy; we relabel a frame
    at distance d ≤ smoothing_window from the boundary with a smoothly
    decaying soft target so the model is not punished for being unsure
    one frame away from the cutoff.

Class balancing is handled at the sampler level — see `make_sampler`.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


@dataclass
class SparseRewardConfig:
    # Window at the end of every episode that counts as "task done".
    terminal_window_frames: int = 15  # 0.5s at 30 fps
    # Soft-label decay window just before the hard boundary (in frames).
    # Frame at d frames before the terminal cutoff gets target ~ 0.5 *
    # (1 - d/smoothing_window). Set to 0 to disable.
    smoothing_window: int = 5

    image_size: int = 224
    # Augmentation knobs.
    color_jitter: tuple = (0.3, 0.3, 0.3, 0.05)  # brightness, contrast, saturation, hue
    random_erasing_p: float = 0.25
    affine_degrees: float = 4.0
    affine_translate: tuple = (0.03, 0.03)
    gaussian_blur_kernel: int = 5
    state_noise_std: float = 0.5  # joint positions are in degrees; ~0.5deg is calibration noise
    state_noise_slider_std: float = 0.05  # last dim is slider.vel
    crop_scale: tuple = (0.85, 1.0)
    crop_ratio: tuple = (0.95, 1.05)

    # The single camera key. Matches the ACT dataset feature name.
    image_key: str = "observation.images.arm_camera"
    state_key: str = "observation.state"


def build_train_transform(cfg: SparseRewardConfig) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(
                cfg.image_size,
                scale=cfg.crop_scale,
                ratio=cfg.crop_ratio,
                antialias=True,
            ),
            T.ColorJitter(*cfg.color_jitter),
            T.RandomApply(
                [T.GaussianBlur(cfg.gaussian_blur_kernel, sigma=(0.1, 1.5))], p=0.3
            ),
            T.RandomAffine(
                degrees=cfg.affine_degrees,
                translate=cfg.affine_translate,
            ),
            T.RandomErasing(p=cfg.random_erasing_p, scale=(0.02, 0.08)),
        ]
    )


def build_eval_transform(cfg: SparseRewardConfig) -> T.Compose:
    return T.Compose(
        [
            T.Resize(cfg.image_size, antialias=True),
            T.CenterCrop(cfg.image_size),
        ]
    )


def _episode_spans(lerobot_dataset) -> dict[int, tuple[int, int, int]]:
    """Return a {episode_index: (dataset_from, dataset_to_exclusive, length)}
    map. Compatible with lerobot v3.0 datasets where `meta.episodes` is an
    HF datasets.Dataset, and falls back to legacy attribute names for
    older versions."""
    episodes = lerobot_dataset.meta.episodes

    if hasattr(episodes, "to_pandas"):  # datasets.Dataset (v3.0)
        df = episodes.to_pandas()
        if "dataset_from_index" in df.columns:
            it = zip(
                df["episode_index"].astype(int),
                df["dataset_from_index"].astype(int),
                df["dataset_to_index"].astype(int),
                df["length"].astype(int),
            )
            return {int(ep): (int(fr), int(to), int(L)) for ep, fr, to, L in it}
        # legacy column name
        if "length" in df.columns:
            spans: dict[int, tuple[int, int, int]] = {}
            cursor = 0
            for ep, L in zip(df["episode_index"].astype(int), df["length"].astype(int)):
                spans[int(ep)] = (cursor, cursor + int(L), int(L))
                cursor += int(L)
            return spans

    if hasattr(episodes, "items"):  # plain dict-of-dicts (legacy)
        spans = {}
        cursor = 0
        for k, v in episodes.items():
            L = int(v["length"])
            spans[int(k)] = (cursor, cursor + L, L)
            cursor += L
        return spans

    raise RuntimeError(
        f"Unable to read episode spans from {type(episodes)}; "
        "lerobot version may be unsupported."
    )


def soft_target(
    frame_index: int,
    episode_length: int,
    cfg: SparseRewardConfig,
) -> float:
    """Hard 1.0 inside terminal window, 0.0 deep before it, smooth ramp in
    a small band just before the boundary."""
    distance_from_end = episode_length - 1 - frame_index  # 0 at last frame
    if distance_from_end < cfg.terminal_window_frames:
        return 1.0
    if cfg.smoothing_window <= 0:
        return 0.0
    d = distance_from_end - cfg.terminal_window_frames  # 0 right outside boundary
    if d >= cfg.smoothing_window:
        return 0.0
    # smooth half-cosine from 0.5 at the boundary edge down to 0.
    return 0.5 * 0.5 * (1.0 + math.cos(math.pi * d / cfg.smoothing_window))


class SparseRewardDataset(Dataset):
    """Wraps a LeRobotDataset, returns (image, state, target)."""

    def __init__(
        self,
        lerobot_dataset,
        cfg: SparseRewardConfig,
        episode_indices: Sequence[int] | None = None,
        transform: T.Compose | None = None,
        train: bool = True,
    ) -> None:
        self.ds = lerobot_dataset
        self.cfg = cfg
        self.train = train
        self.transform = transform or (
            build_train_transform(cfg) if train else build_eval_transform(cfg)
        )

        self._episode_spans = _episode_spans(lerobot_dataset)

        if episode_indices is None:
            episode_indices = sorted(self._episode_spans.keys())
        self.allowed_episodes = set(int(e) for e in episode_indices)

        # Build a flat index of global frame ids that survive the episode
        # filter. LeRobotDataset stores frames flatly, addressed by a
        # global index.
        self._global_ids: list[int] = []
        self._labels: list[float] = []
        for ep in sorted(self.allowed_episodes):
            fr, to, length = self._episode_spans[ep]
            assert to - fr == length, (
                f"episode {ep}: meta length {length} != index span {to - fr}"
            )
            for local in range(length):
                self._global_ids.append(fr + local)
                self._labels.append(soft_target(local, length, cfg))

        self._labels_np = np.asarray(self._labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._global_ids)

    @property
    def labels(self) -> np.ndarray:
        return self._labels_np

    def positive_fraction(self) -> float:
        return float((self._labels_np >= 0.5).mean())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_id = self._global_ids[idx]
        sample = self.ds[global_id]

        image = sample[self.cfg.image_key]  # (C, H, W) float in [0, 1] or uint8
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        image = self.transform(image)

        state = sample[self.cfg.state_key].float().clone()
        if self.train and self.cfg.state_noise_std > 0:
            noise = torch.randn_like(state)
            noise[..., :-1] *= self.cfg.state_noise_std
            noise[..., -1] *= self.cfg.state_noise_slider_std
            state = state + noise

        target = torch.tensor(self._labels[idx], dtype=torch.float32)
        return {"image": image, "state": state, "target": target}


def make_balanced_sampler(
    dataset: SparseRewardDataset,
    positive_oversample: float = 1.0,
) -> WeightedRandomSampler:
    """WeightedRandomSampler that approximately balances positive vs
    negative frames per epoch. `positive_oversample > 1` skews the mix
    toward positives."""
    labels = (dataset.labels >= 0.5).astype(np.float64)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        weights = np.ones_like(labels)
    else:
        w_pos = (0.5 / n_pos) * positive_oversample
        w_neg = 0.5 / n_neg
        weights = labels * w_pos + (1.0 - labels) * w_neg
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def episode_train_val_split(
    all_episodes: Sequence[int],
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Hold out whole episodes for validation, never frames within an
    episode (terminal frames in a held-out episode are the only true test
    of generalization)."""
    rng = random.Random(seed)
    episodes = list(all_episodes)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val = sorted(episodes[:n_val])
    train = sorted(episodes[n_val:])
    return train, val


def load_lerobot_dataset(
    repo_id: str,
    root: str | Path | None = None,
    image_keys: Sequence[str] | None = None,
):
    """Thin wrapper that imports lerobot lazily so the model file stays
    importable without the heavy dependency."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {}
    if root is not None:
        kwargs["root"] = str(root)
    if image_keys is not None:
        # lerobot 0.5+ accepts `video_backend` and `tolerance_s`; image
        # filtering is implicit in `delta_timestamps={}` (no temporal
        # stacking).
        pass
    kwargs.setdefault("video_backend", "torchcodec")
    return LeRobotDataset(repo_id, **kwargs)
