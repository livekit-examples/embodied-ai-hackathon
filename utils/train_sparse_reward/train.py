#!/usr/bin/env python
"""Train a sparse-reward classifier on a LeRobotDataset.

Inputs match ACT (single camera frame + 7-d state, no temporal buffer).
Targets: 1 inside the last `terminal_window_frames` of each episode,
else 0, with a small smoothed transition band.

Configure via env vars (mirrors the ACT trainer's style):

  DATASET_REPO_ID   HF or local repo id (e.g. binhpham/spc-pick-sharpie).
                    If not on the hub, set DATASET_ROOT to its parent dir
                    on disk (lerobot expects `<root>/<repo_id>/meta/...`).
  DATASET_ROOT      Optional local root (passed to LeRobotDataset.root).
  RUN_NAME          Output run name (default: reward-<dataset>-<date>).
  NUM_EPOCHS        Training epochs (default 20).
  BATCH_SIZE        (default 64)
  LR                AdamW LR (default 3e-4)
  TERMINAL_WINDOW   Frames at the end of an episode that count as success
                    (default 15, i.e. 0.5s @ 30fps).
  VAL_FRACTION      Episode-level holdout (default 0.15).
  OUTPUT_ROOT       Parent dir for run outputs (default ./outputs).
  POSITIVE_OVERSAMPLE  How aggressively to oversample positives in
                       training batches (default 1.0 = balanced).
  IMAGE_SIZE        Square image size fed to the model (default 224).
  NUM_WORKERS       DataLoader workers (default 8).
  SEED              (default 0)

Resumes from `<OUTPUT_ROOT>/<RUN_NAME>/last.pt` if it exists.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the package directory is on path when invoked as `python train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    SparseRewardConfig,
    SparseRewardDataset,
    build_eval_transform,
    build_train_transform,
    episode_train_val_split,
    load_lerobot_dataset,
    make_balanced_sampler,
)
from reward_model import RewardModelConfig, SparseRewardModel  # noqa: E402


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(
    model: SparseRewardModel,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(img, state)
        all_logits.append(logits.float().cpu())
        all_targets.append(tgt.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits)
    hard = (targets >= 0.5).float()
    pred = (probs >= threshold).float()

    tp = ((pred == 1) & (hard == 1)).sum().item()
    fp = ((pred == 1) & (hard == 0)).sum().item()
    fn = ((pred == 0) & (hard == 1)).sum().item()
    tn = ((pred == 0) & (hard == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    # AUROC, computed manually to avoid sklearn dependency.
    auroc = _auroc(probs.numpy(), hard.numpy())

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    ).item()

    return {
        "loss": bce,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "positive_rate": float(hard.mean().item()),
        "predicted_positive_rate": float(pred.mean().item()),
    }


def _state_dim_from_dataset(lerobot_ds, state_key: str) -> int:
    """Read the state feature dim from whichever attribute the installed
    lerobot version exposes."""
    features = getattr(lerobot_ds, "features", None) or getattr(
        lerobot_ds.meta, "features", {}
    )
    feat = features[state_key]
    if "shape" in feat:
        return int(feat["shape"][0])
    return int(len(feat["names"]))


def _auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    pos = scores[targets >= 0.5]
    neg = scores[targets < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney U via ranking.
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main() -> None:
    seed = env_int("SEED", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_repo = env_str("DATASET_REPO_ID", "binhpham/spc-pick-sharpie")
    dataset_root = os.environ.get("DATASET_ROOT")
    num_epochs = env_int("NUM_EPOCHS", 20)
    batch_size = env_int("BATCH_SIZE", 64)
    lr = env_float("LR", 3e-4)
    weight_decay = env_float("WEIGHT_DECAY", 1e-4)
    terminal_window = env_int("TERMINAL_WINDOW", 15)
    val_fraction = env_float("VAL_FRACTION", 0.15)
    output_root = Path(env_str("OUTPUT_ROOT", "./outputs"))
    positive_oversample = env_float("POSITIVE_OVERSAMPLE", 1.0)
    image_size = env_int("IMAGE_SIZE", 224)
    num_workers = env_int("NUM_WORKERS", 8)

    safe_repo = dataset_repo.replace("/", "_")
    run_name = env_str(
        "RUN_NAME",
        f"reward-{safe_repo}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"[train] device={device} run={run_name} output_dir={output_dir}")

    print(f"[train] loading LeRobotDataset {dataset_repo} (root={dataset_root}) ...")
    lerobot_ds = load_lerobot_dataset(dataset_repo, root=dataset_root)
    print(
        f"[train] dataset: {len(lerobot_ds)} frames, "
        f"{lerobot_ds.num_episodes} episodes, fps={lerobot_ds.fps}"
    )

    ds_cfg = SparseRewardConfig(
        terminal_window_frames=terminal_window,
        image_size=image_size,
    )

    all_episodes = list(range(int(lerobot_ds.num_episodes)))
    train_eps, val_eps = episode_train_val_split(
        all_episodes, val_fraction=val_fraction, seed=seed
    )
    print(f"[train] split: {len(train_eps)} train / {len(val_eps)} val episodes")

    train_ds = SparseRewardDataset(
        lerobot_ds, ds_cfg, episode_indices=train_eps, train=True
    )
    val_ds = SparseRewardDataset(
        lerobot_ds, ds_cfg, episode_indices=val_eps, train=False
    )
    print(
        f"[train] frames: train={len(train_ds)} ({train_ds.positive_fraction():.3%} positive) "
        f"val={len(val_ds)} ({val_ds.positive_fraction():.3%} positive)"
    )

    sampler = make_balanced_sampler(train_ds, positive_oversample=positive_oversample)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
    )

    state_dim = _state_dim_from_dataset(lerobot_ds, ds_cfg.state_key)
    model_cfg = RewardModelConfig(state_dim=state_dim, image_size=image_size)
    model = SparseRewardModel(model_cfg).to(device)
    print(
        f"[train] model params: "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"total={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    bce = torch.nn.BCEWithLogitsLoss()

    last_ckpt = output_dir / "last.pt"
    best_ckpt = output_dir / "best.pt"
    start_epoch = 0
    best_auroc = -math.inf
    if last_ckpt.is_file():
        print(f"[train] resuming from {last_ckpt}")
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_auroc = state.get("best_auroc", -math.inf)

    cfg_blob = {
        "dataset_repo_id": dataset_repo,
        "dataset_root": dataset_root,
        "ds_cfg": asdict(ds_cfg),
        "model_cfg": asdict(model_cfg),
        "train": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "terminal_window": terminal_window,
            "val_fraction": val_fraction,
            "positive_oversample": positive_oversample,
            "image_size": image_size,
            "seed": seed,
        },
        "train_episodes": train_eps,
        "val_episodes": val_eps,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg_blob, indent=2))

    log_path = output_dir / "log.jsonl"
    log_fp = log_path.open("a")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(img, state)
                loss = bce(logits, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * img.size(0)
            running_n += img.size(0)

            if step % 50 == 0:
                print(
                    f"[train] epoch {epoch:>3} step {step:>4}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss = running_loss / max(running_n, 1)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"[train] epoch {epoch:>3} done in {dt:.1f}s  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  "
            f"val_f1={val_metrics['f1']:.3f}  "
            f"val_auroc={val_metrics['auroc']:.3f}"
        )

        log_fp.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "wall_time_s": dt,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            + "\n"
        )
        log_fp.flush()

        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_auroc": best_auroc,
            "model_cfg": asdict(model_cfg),
            "ds_cfg": asdict(ds_cfg),
        }
        torch.save(state_dict, last_ckpt)

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            state_dict["best_auroc"] = best_auroc
            torch.save(state_dict, best_ckpt)
            print(f"[train] new best AUROC={best_auroc:.3f} → {best_ckpt}")

    log_fp.close()
    print(f"[train] done. best AUROC={best_auroc:.3f}; checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
