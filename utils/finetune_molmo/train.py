#!/usr/bin/env python
"""Fine-tune MolmoAct2-SO100_101 on a portal-hitl LeRobotDataset.

Sibling of `utils/data_collection/policies/act/train.py`. Same dataset,
same dataloader/resume patches, same env-var knobs. The differences are
all MolmoAct2-specific:

  * the released SO-100/101 checkpoint expects a 6-DOF arm state/action;
    our rig records a 7-dim state/action (six SO-101 joint values plus
    `slider.vel`). We slice the 7th dim out of both observation.state
    and action before the trainer sees them.
  * the model was pre-trained on the canonical LeRobot v3.0 calibration
    convention; our rig records v2.1 raw values. We apply the same
    `signs * x + offsets` transform that `operators/molmo/molmo_operator.py`
    uses at inference, so the closed loop sees one consistent frame.
  * MolmoAct2 wants quantile stats. The recorded dataset only ships
    mean/std, so we tell the policy to use MEAN_STD normalization for
    STATE and ACTION (per the official MolmoAct2 LeRobot docs).
  * the trainer entry point is `lerobot.scripts.lerobot_train` with
    `--policy.type=molmoact2` and `--policy.checkpoint_path=...` to
    start from the released HF weights.

Configure via env:
  DATASET_REPO_ID      HF or local repo id (default local/portal-hitl)
  RUN_NAME             Output run name prefix
  NUM_STEPS            Total training steps (default 10000)
  BATCH_SIZE           Per-GPU batch size (default 8)
  LR                   VLM LR (default 1e-5)
  ACTION_EXPERT_LR     Action-expert LR (default 5e-5)
  CHUNK_SIZE           Action horizon (default 30, matches the SO-100/101
                       released checkpoint's action_horizon=30)
  IMAGE_KEYS           JSON list of observation.images.* keys. If unset,
                       we auto-discover from the dataset's info.json and
                       prefer (overhead_camera, arm_camera) order to
                       match the molmo operator's top/side mapping.
  CHECKPOINT_PATH      Starting HF checkpoint (default allenai/MolmoAct2-SO100_101)
  MODEL_DTYPE          bfloat16 / float16 / float32 (default bfloat16)
  TRAIN_ACTION_EXPERT_ONLY    Set to 1 to freeze the VLM/vision tower
  ENABLE_LORA_VLM             Set to 1 to LoRA-adapt the VLM
  OUTPUT_ROOT          Parent dir for run outputs (default /outputs —
                       the SkyPilot mount path; override on plain servers)
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# DataLoader tuning patch, lerobot hardcodes prefetch_factor=2 and
# doesn't expose persistent_workers; force both for warm worker reuse.
_orig_dl_init = torch.utils.data.DataLoader.__init__
def _dl_init_tuned(self, *args, **kwargs):
    if kwargs.get("num_workers", 0):
        kwargs.setdefault("persistent_workers", True)
        kwargs["prefetch_factor"] = 4
    _orig_dl_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _dl_init_tuned


# Object-storage FUSE mounts (S3 / Nebius) don't support symlinks; lerobot
# tries to maintain `checkpoints/last -> NNNNNN/` after every save. Catch
# the resulting OSError and warn instead of crashing.
#
# PyPI lerobot exposes this as `lerobot.utils.train_utils`; the allenai
# `molmoact2-policy` fork keeps it at `lerobot.common.train_utils`. Try
# both so the same trainer runs against either install.
try:
    import lerobot.utils.train_utils as _train_utils
except ModuleNotFoundError:
    import lerobot.common.train_utils as _train_utils
_orig_update_last = _train_utils.update_last_checkpoint
def _safe_update_last_checkpoint(checkpoint_dir):
    try:
        return _orig_update_last(checkpoint_dir)
    except OSError as e:
        print(f"[finetune-molmo] skipping 'last' symlink (fs unsupported): {e}")
_train_utils.update_last_checkpoint = _safe_update_last_checkpoint


# 6-DOF SO-101 calibration: rig (v2.1) → policy (v3.0). Same constants
# as operators/molmo/molmo_operator.py. Index order matches the dataset's
# `observation.state.names[:6]`. Applied in both directions:
#   training state/action: x_policy = signs * x_rig + offsets
#   inference action:      x_rig    = (x_policy - offsets) * signs
JOINT_SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
JOINT_OFFSETS_DEG = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_DIM = 6  # drop slider.vel (dim 6 of the 7-dim record)

# Torch versions of the transform for in-loop application.
_JOINT_SIGNS_T = torch.from_numpy(JOINT_SIGNS)
_JOINT_OFFSETS_T = torch.from_numpy(JOINT_OFFSETS_DEG)


# Patch LeRobotDataset so MolmoAct2 sees a 6-dim policy-frame robot. Three
# surfaces have to agree: meta.features (declared shape + names),
# meta.stats (normalization constants), and __getitem__ (the actual
# tensors). If any one of them is still 7-dim or in the rig frame, the
# trainer will either crash with a shape mismatch or, worse, train with
# stats that don't match the data.
import lerobot.datasets.lerobot_dataset as _lds


def _transform_stat(stat: np.ndarray, kind: str) -> np.ndarray:
    """Apply `signs * x + offsets` to a per-dim stat vector after
    slicing to ARM_DIM. For sign-sensitive stats (min/max/q01/q99),
    flip the pair where the sign is negative."""
    arr = np.asarray(stat, dtype=np.float32)
    if arr.shape[-1] < ARM_DIM:
        return arr
    arr = arr[..., :ARM_DIM]
    if kind in ("mean",):
        return JOINT_SIGNS * arr + JOINT_OFFSETS_DEG
    if kind in ("std",):
        # |±1| = 1, so std is unchanged. Still drop slider.
        return arr
    if kind in ("min", "q01"):
        # When sign is negative, this stat's role becomes max/q99 (sort
        # order flips). The companion stat handles the other direction.
        return JOINT_SIGNS * arr + JOINT_OFFSETS_DEG
    if kind in ("max", "q99"):
        return JOINT_SIGNS * arr + JOINT_OFFSETS_DEG
    # Unknown kind — slice only, leave the values alone.
    return arr


def _patch_feature_meta(meta) -> None:
    """Slice observation.state / action features and stats down to 6
    dims, and transform stats from rig to policy frame. Called once per
    dataset construction."""
    features = getattr(meta, "features", None)
    if isinstance(features, dict):
        for key in ("observation.state", "action"):
            f = features.get(key)
            if not f:
                continue
            shape = f.get("shape")
            if shape and tuple(shape) == (7,):
                f["shape"] = [ARM_DIM]
                names = f.get("names")
                if isinstance(names, list) and len(names) == 7:
                    f["names"] = names[:ARM_DIM]

    stats = getattr(meta, "stats", None)
    if isinstance(stats, dict):
        for key in ("observation.state", "action"):
            block = stats.get(key)
            if not isinstance(block, dict):
                continue
            for stat_name in list(block.keys()):
                val = block[stat_name]
                # Some stat values are scalars (e.g. count); skip those.
                arr = np.asarray(val)
                if arr.ndim == 0 or arr.shape[-1] != 7:
                    continue
                # For min/max and q01/q99, the negative-sign joint gets
                # its pair swapped before the linear transform. Easier
                # to do that in a single pass over both stats.
            # Pair-swap pass for min↔max and q01↔q99 where signs<0.
            for lo_key, hi_key in (("min", "max"), ("q01", "q99")):
                lo = block.get(lo_key)
                hi = block.get(hi_key)
                if lo is None or hi is None:
                    continue
                lo_arr = np.asarray(lo, dtype=np.float32)
                hi_arr = np.asarray(hi, dtype=np.float32)
                if lo_arr.shape[-1] != 7 or hi_arr.shape[-1] != 7:
                    continue
                lo_sliced = lo_arr[:ARM_DIM].copy()
                hi_sliced = hi_arr[:ARM_DIM].copy()
                flip = JOINT_SIGNS < 0
                lo_new = np.where(flip, hi_sliced, lo_sliced)
                hi_new = np.where(flip, lo_sliced, hi_sliced)
                block[lo_key] = JOINT_SIGNS * lo_new + JOINT_OFFSETS_DEG
                block[hi_key] = JOINT_SIGNS * hi_new + JOINT_OFFSETS_DEG
            # Mean / std are easy: just transform mean, keep std.
            for stat_name in ("mean", "std"):
                val = block.get(stat_name)
                if val is None:
                    continue
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape[-1] != 7:
                    continue
                block[stat_name] = _transform_stat(arr, stat_name)


_orig_ds_init = _lds.LeRobotDataset.__init__
def _patched_ds_init(self, *args, **kwargs):
    _orig_ds_init(self, *args, **kwargs)
    _patch_feature_meta(self.meta)
_lds.LeRobotDataset.__init__ = _patched_ds_init


_orig_getitem = _lds.LeRobotDataset.__getitem__
def _patched_getitem(self, idx):
    item = _orig_getitem(self, idx)
    for key in ("observation.state", "action"):
        if key not in item:
            continue
        t = item[key]
        if not isinstance(t, torch.Tensor):
            continue
        if t.shape[-1] != 7:
            continue
        sliced = t[..., :ARM_DIM].to(torch.float32)
        signs = _JOINT_SIGNS_T.to(sliced.device)
        offs = _JOINT_OFFSETS_T.to(sliced.device)
        item[key] = signs * sliced + offs
    return item
_lds.LeRobotDataset.__getitem__ = _patched_getitem


# Env / paths
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "local/portal-hitl")
DATASET_ROOT = os.environ.get("DATASET_ROOT")  # optional, for local datasets
NUM_STEPS = int(os.environ.get("NUM_STEPS", 10000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
LR = os.environ.get("LR", "1e-5")
ACTION_EXPERT_LR = os.environ.get("ACTION_EXPERT_LR", "5e-5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 30))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "allenai/MolmoAct2-SO100_101")
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "bfloat16")
SETUP_TYPE = os.environ.get("SETUP_TYPE", "single so100/so101 robotic arm in molmoact2")
CONTROL_MODE = os.environ.get("CONTROL_MODE", "absolute joint pose")
TRAIN_ACTION_EXPERT_ONLY = os.environ.get("TRAIN_ACTION_EXPERT_ONLY", "0") == "1"
ENABLE_LORA_VLM = os.environ.get("ENABLE_LORA_VLM", "0") == "1"

_base_name = os.environ.get("RUN_NAME", "molmoact2-leslider")
RUN_NAME = f"{_base_name}_steps{NUM_STEPS}_bs{BATCH_SIZE}_lr{LR}_chunk{CHUNK_SIZE}"

_save_eval_freq = max(NUM_STEPS // 4, 1000)

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/outputs"))
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME


def _resolve_image_keys() -> list[str]:
    """Pick the image keys to feed MolmoAct2. Honor IMAGE_KEYS env if set,
    else auto-discover from the dataset's info.json. Prefers overhead +
    arm order to match the molmo operator's top/side mapping."""
    raw = os.environ.get("IMAGE_KEYS")
    if raw:
        keys = json.loads(raw)
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise SystemExit(
                f"IMAGE_KEYS must be a JSON list of strings; got: {raw!r}"
            )
        return keys

    # Local dataset: read info.json directly.
    candidates: list[Path] = []
    if DATASET_ROOT:
        candidates.append(Path(DATASET_ROOT) / "meta" / "info.json")
    candidates.append(
        Path(__file__).resolve().parents[1]
        / "data_collection" / "data" / DATASET_REPO_ID / "meta" / "info.json"
    )
    info_path = next((p for p in candidates if p.is_file()), None)
    if info_path is None:
        # Remote / HF dataset — caller must provide IMAGE_KEYS explicitly.
        raise SystemExit(
            "IMAGE_KEYS env var is required when the dataset is not on disk. "
            "Example: IMAGE_KEYS='[\"observation.images.overhead_camera\","
            "\"observation.images.arm_camera\"]'"
        )
    info = json.loads(info_path.read_text())
    image_keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    if not image_keys:
        raise SystemExit(f"no observation.images.* features in {info_path}")
    # Stable preference: overhead first (top), arm second (side).
    pref = ("observation.images.overhead_camera", "observation.images.arm_camera")
    ordered = [k for k in pref if k in image_keys] + [k for k in image_keys if k not in pref]
    return ordered


IMAGE_KEYS = _resolve_image_keys()


def _assert_writable(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"[finetune-molmo] cannot write to OUTPUT_DIR={path}: {exc}\n"
            f"  set OUTPUT_ROOT to a writable path, e.g.\n"
            f"    OUTPUT_ROOT=$HOME/outputs uv run python train.py\n"
        )


_assert_writable(OUTPUT_DIR)


def _find_resume_config(output_dir: Path) -> Path | None:
    """Highest valid checkpoint without relying on the `last/` symlink."""
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    candidates = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for c in candidates:
        cfg = c / "pretrained_model" / "train_config.json"
        state = c / "training_state" / "training_step.json"
        if cfg.is_file() and state.is_file():
            return cfg
    return None


_resume_cfg = _find_resume_config(OUTPUT_DIR)
if _resume_cfg is not None:
    print(f"[finetune-molmo] resuming from {_resume_cfg.parent.parent}")
    sys.argv[1:] = [
        "--resume=true",
        f"--config_path={_resume_cfg}",
    ]
else:
    if OUTPUT_DIR.is_dir():
        print(f"[finetune-molmo] no valid checkpoint under {OUTPUT_DIR}; clearing for fresh run")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    args = [
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={CHECKPOINT_PATH}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--dataset.video_backend=torchcodec",
        "--dataset.image_transforms.enable=true",
        f"--output_dir={OUTPUT_DIR}",
        "--policy.device=cuda",
        f"--policy.model_dtype={MODEL_DTYPE}",
        "--policy.action_mode=continuous",
        f"--policy.chunk_size={CHUNK_SIZE}",
        f"--policy.n_action_steps={CHUNK_SIZE}",
        f"--policy.setup_type={SETUP_TYPE}",
        f"--policy.control_mode={CONTROL_MODE}",
        f"--policy.image_keys={json.dumps(IMAGE_KEYS)}",
        "--policy.num_flow_timesteps=8",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_embedding=true",
        "--policy.normalize_gripper=false",
        "--policy.enable_knowledge_insulation=false",
        f"--policy.optimizer_lr={LR}",
        f"--policy.optimizer_action_expert_lr={ACTION_EXPERT_LR}",
        # Recorded datasets only ship mean/std stats. MEAN_STD keeps the
        # MolmoAct2 docs' alternative normalization path working without
        # requiring an offline `augment_dataset_quantile_stats.py` pass.
        "--policy.normalization_mapping={\"ACTION\": \"MEAN_STD\", \"STATE\": \"MEAN_STD\", \"VISUAL\": \"IDENTITY\"}",
        "--policy.push_to_hub=false",
        f"--steps={NUM_STEPS}",
        f"--batch_size={BATCH_SIZE}",
        "--num_workers=4",
        "--tolerance_s=0.5",
        "--log_freq=20",
        f"--save_freq={_save_eval_freq}",
        f"--eval_freq=-1",
    ]
    if DATASET_ROOT:
        args.append(f"--dataset.root={DATASET_ROOT}")
    if TRAIN_ACTION_EXPERT_ONLY:
        args.append("--policy.train_action_expert_only=true")
    if ENABLE_LORA_VLM:
        args.append("--policy.enable_lora_vlm=true")
    sys.argv[1:] = args

runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True)
