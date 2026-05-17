# train_sparse_reward

Train a sparse-reward classifier on a portal-hitl LeRobotDataset. The
model sees the **same single-step inputs as ACT** — one camera frame
(`observation.images.arm_camera`) plus the 7-d proprio state
(`observation.state`) — with **no temporal buffer**. It outputs a single
logit; `sigmoid(logit)` ≈ P(task is in its final ~0.5 s window).

Use it as a learned "task done" signal next to ACT inference, or as a
sparse reward for offline RL.

## Labels

For each episode of length `L`, frames `[L - 15, L)` are labelled `1`,
the rest `0`. At 30 fps that is the last 0.5 s. The boundary is fuzzy
in practice, so frames just before the cutoff get smoothly decaying
soft targets (configurable; set `SparseRewardConfig.smoothing_window=0`
to disable).

## Augmentation

- Image: random resized crop, color jitter, gaussian blur, random
  affine, random erasing. **No horizontal flip** — flipping would
  invert gripper geometry.
- State: small Gaussian noise (≈ 0.5 deg on joints, 0.05 on
  `slider.vel`).
- Soft label smoothing across the terminal boundary.
- Class balance via `WeightedRandomSampler` so each batch is ≈ 50 / 50
  positive / negative. Without this, positives are ~6 % of frames and
  the model collapses to "always 0".

Episode-level train / val split: whole episodes are held out, never
frames within an episode. Memorising 15 frames inside a seen episode
is trivial; generalising to a new episode's terminal window is the
real test.

## Train

From the repo root:

```bash
cd utils/train_sparse_reward
uv sync
DATASET_REPO_ID=binhpham/spc-pick-sharpie \
DATASET_ROOT=../data_collection/data \
uv run python train.py
```

The `DATASET_ROOT` env var points at the parent directory holding the
LeRobotDataset on disk; omit it if loading from the HF hub.

Other knobs (all env vars, defaults in parens):

| var | default | notes |
| --- | --- | --- |
| `NUM_EPOCHS` | 20 | |
| `BATCH_SIZE` | 64 | |
| `LR` | 3e-4 | AdamW + OneCycle |
| `WEIGHT_DECAY` | 1e-4 | |
| `TERMINAL_WINDOW` | 15 | frames at end-of-episode that count as success |
| `VAL_FRACTION` | 0.15 | episode-level holdout |
| `POSITIVE_OVERSAMPLE` | 1.0 | > 1 skews batches toward positives |
| `IMAGE_SIZE` | 224 | model input size |
| `NUM_WORKERS` | 8 | DataLoader workers |
| `OUTPUT_ROOT` | `./outputs` | |
| `RUN_NAME` | auto | |
| `SEED` | 0 | |

Outputs land under `<OUTPUT_ROOT>/<RUN_NAME>/`:

```
last.pt          # latest checkpoint (resume target)
best.pt          # best val AUROC so far
config.json      # full run config + episode split
log.jsonl        # per-epoch metrics
```

Resuming is automatic if `last.pt` exists.

## Inference

```python
from inference import RewardScorer

scorer = RewardScorer.from_checkpoint("outputs/<run>/best.pt")
prob = scorer.score(rgb_uint8_hxwx3, state_7_floats)  # → P(done)
```

For live use alongside ACT, wrap a small moving-average filter on
`prob` and trigger "done" when it crosses, say, 0.7 for ≥ 10 ticks.

## Why these choices

- **Single-step input, no buffer**: matches ACT's interface so the same
  observation tap drives both the policy and the reward. No
  observation history means lower memory and no buffer-warmup issues at
  episode start.
- **ResNet18 + small MLP**: a few hundred MB, trains in minutes on a
  single GPU for ~50 episodes. Larger backbones overfit fast on this
  data scale.
- **WeightedRandomSampler over class weight in BCE**: each step still
  sees both classes; pure `pos_weight` on a heavily imbalanced batch
  produces noisy gradients.
- **Episode-level holdout**: per-frame holdout would let the model
  cheat by memorising the same episode it later sees in val.
