#!/usr/bin/env python
"""Train ACT on a portal-hitl-recorded leslider dataset.

Unlike the leslider repo's `policies/act/train.py`, this script does
*not* slice `observation.state`, portal-hitl records a 7-dim state
natively (six arm `.pos` plus `slider.vel`, no motor currents), so the
training data is already at the right shape.

What we keep from leslider:
  * Symlink-safe `update_last_checkpoint` shim for object-storage FUSE
    mounts (S3/Nebius), which don't support symlinks.
  * `persistent_workers=True` + `prefetch_factor=4` patch on the
    DataLoader (lerobot hardcodes `prefetch_factor=2`).
  * Resume protocol that doesn't need the `checkpoints/last/` symlink.

Configure via env:
  DATASET_REPO_ID    HF or local repo id of the recorded dataset
  RUN_NAME           Output run name (defaults to algo + date)
  NUM_STEPS          Total training steps (default 20k)
  BATCH_SIZE         (default 64)
  LR                 Optimizer LR (default 5e-5)
  CHUNK_SIZE         ACT chunk_size + n_action_steps (default 50)
  OUTPUT_ROOT        Parent dir for run outputs (default /outputs — the
                     SkyPilot mount path; override on plain servers)
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

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
import lerobot.utils.train_utils as _train_utils
_orig_update_last = _train_utils.update_last_checkpoint
def _safe_update_last_checkpoint(checkpoint_dir):
    try:
        return _orig_update_last(checkpoint_dir)
    except OSError as e:
        print(f"[train.py] skipping 'last' symlink (fs unsupported): {e}")
_train_utils.update_last_checkpoint = _safe_update_last_checkpoint


DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "local/portal-hitl")
NUM_STEPS = int(os.environ.get("NUM_STEPS", 20000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
LR = os.environ.get("LR", "5e-5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 50))

_base_name = os.environ.get("RUN_NAME", "act-leslider")
RUN_NAME = f"{_base_name}_steps{NUM_STEPS}_bs{BATCH_SIZE}_lr{LR}_chunk{CHUNK_SIZE}"

_save_eval_freq = max(NUM_STEPS // 4, 1000)

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/outputs"))
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME


def _assert_writable(path: Path) -> None:
    """Fail fast if we can't write to the output tree. Lerobot only opens
    the checkpoint dir at the first save_freq tick (5000+ steps in), so a
    permission error there costs an hour of training. Catch it now."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        msg = (
            f"[train.py] cannot write to OUTPUT_DIR={path}: {exc}\n"
            f"  set OUTPUT_ROOT to a writable path, e.g.\n"
            f"    OUTPUT_ROOT=$HOME/outputs uv run python policies/act/train.py\n"
        )
        raise SystemExit(msg)


_assert_writable(OUTPUT_DIR)


def _find_resume_config(output_dir: Path) -> Path | None:
    """Find the highest *valid* checkpoint without relying on the `last/`
    symlink. Skips checkpoints whose pretrained_model/training_state are
    partial, those represent runs killed mid-save."""
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
    print(f"[train.py] resuming from {_resume_cfg.parent.parent}")
    sys.argv[1:] = [
        "--resume=true",
        f"--config_path={_resume_cfg}",
    ]
else:
    if OUTPUT_DIR.is_dir():
        print(f"[train.py] no valid checkpoint under {OUTPUT_DIR}; clearing for fresh run")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    sys.argv[1:] = [
        "--policy.type=act",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--dataset.video_backend=torchcodec",
        f"--output_dir={OUTPUT_DIR}",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        f"--policy.chunk_size={CHUNK_SIZE}",
        f"--policy.n_action_steps={CHUNK_SIZE}",
        f"--policy.optimizer_lr={LR}",
        "--policy.optimizer_lr_backbone=1e-5",
        f"--steps={NUM_STEPS}",
        f"--batch_size={BATCH_SIZE}",
        "--num_workers=16",
        "--tolerance_s=0.5",
        "--log_freq=100",
        f"--save_freq={_save_eval_freq}",
        f"--eval_freq={_save_eval_freq}",
        "--policy.push_to_hub=false",
    ]

runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True)
