#!/usr/bin/env python
"""Train Diffusion Policy on a portal-hitl-recorded leslider dataset.

Same shape as `policies/act/train.py`: no state mask (portal-hitl records
7-dim natively), keeps the leslider DataLoader/symlink shims.

Configure via env:
  DATASET_REPO_ID    HF or local repo id of the recorded dataset
  RUN_NAME           Output run name
  NUM_STEPS          Total training steps (default 100k)
  BATCH_SIZE         (default 64)
  LR                 Optimizer LR (default 1e-4)
  HORIZON            Diffusion horizon (default 16)
  N_ACTION_STEPS     Actions executed per replan (default 8)
  N_OBS_STEPS        Observation history length (default 2)
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

import torch

# Force persistent_workers + larger prefetch on the lerobot DataLoader.
_orig_dl_init = torch.utils.data.DataLoader.__init__
def _dl_init_tuned(self, *args, **kwargs):
    if kwargs.get("num_workers", 0):
        kwargs.setdefault("persistent_workers", True)
        kwargs["prefetch_factor"] = 4
    _orig_dl_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _dl_init_tuned


# Object-storage FUSE mounts don't support symlinks; swallow the error
# from lerobot's update_last_checkpoint instead of crashing the run.
import lerobot.utils.train_utils as _train_utils
_orig_update_last = _train_utils.update_last_checkpoint
def _safe_update_last_checkpoint(checkpoint_dir):
    try:
        return _orig_update_last(checkpoint_dir)
    except OSError as e:
        print(f"[train.py] skipping 'last' symlink (fs unsupported): {e}")
_train_utils.update_last_checkpoint = _safe_update_last_checkpoint


DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "local/portal-hitl")
NUM_STEPS = int(os.environ.get("NUM_STEPS", 100000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
LR = os.environ.get("LR", "1e-4")
HORIZON = int(os.environ.get("HORIZON", 16))
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", 8))
N_OBS_STEPS = int(os.environ.get("N_OBS_STEPS", 2))

_base_name = os.environ.get("RUN_NAME", "diffusion-leslider")
RUN_NAME = f"{_base_name}_steps{NUM_STEPS}_bs{BATCH_SIZE}_lr{LR}_h{HORIZON}_na{N_ACTION_STEPS}"

_save_eval_freq = max(NUM_STEPS // 4, 1000)

OUTPUT_DIR = Path(f"/outputs/{RUN_NAME}")


def _find_resume_config(output_dir: Path) -> Path | None:
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
        "--policy.type=diffusion",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--dataset.video_backend=torchcodec",
        f"--output_dir={OUTPUT_DIR}",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        "--policy.compile_model=true",
        f"--policy.horizon={HORIZON}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        f"--policy.n_obs_steps={N_OBS_STEPS}",
        "--policy.use_group_norm=true",
        "--policy.use_separate_rgb_encoder_per_camera=false",
        f"--policy.optimizer_lr={LR}",
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
