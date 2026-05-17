# finetune_molmo

Fine-tune `allenai/MolmoAct2-SO100_101` on a portal-hitl LeRobotDataset —
the same dataset shape the ACT trainer eats. Standalone uv project; lives
next to `utils/data_collection/` but does not import from it.

This is the MolmoAct2 sibling of `utils/data_collection/policies/act/train.py`.
Same env-var knobs, same resume protocol, same DataLoader/checkpoint
patches. The MolmoAct2-specific differences are documented at the top of
`train.py`:

  * the dataset's 7-dim state/action (six SO-101 joints + `slider.vel`)
    is sliced down to 6 dims before training,
  * the rig's v2.1 raw calibration is mapped to MolmoAct2's v3.0
    convention with the same `signs * x + offsets` transform the molmo
    operator uses at inference,
  * the trainer is `lerobot.scripts.lerobot_train` with
    `--policy.type=molmoact2` and `--policy.checkpoint_path=allenai/MolmoAct2-SO100_101`.

## Setup

Needs a CUDA box. Fine-tuning the action expert only fits in a single
H100 80GB at `batch_size=8` (16 GiB peak per the MolmoAct2 docs). Full
fine-tune is ~48 GiB at `batch_size=8`.

```bash
uv sync
```

The dep `lerobot[molmoact2]` resolves to the
`allenai/lerobot@molmoact2-policy` fork, which is the only branch with
the MolmoAct2 policy ported into LeRobot's training loop.

## Train

```bash
DATASET_REPO_ID=binhpham/spc-pick-sharpie \
DATASET_ROOT=$(pwd)/../data_collection/data/binhpham/spc-pick-sharpie \
OUTPUT_ROOT=$HOME/outputs \
uv run python train.py
```

If the dataset is on HuggingFace, drop `DATASET_ROOT` and set
`IMAGE_KEYS` explicitly (we can't read the image-key list from
`info.json` without the local dataset).

```bash
DATASET_REPO_ID=you/your-dataset \
IMAGE_KEYS='["observation.images.overhead_camera","observation.images.arm_camera"]' \
uv run python train.py
```

## Env vars

| Var                       | Default                                  | Notes                                                                 |
| ------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `DATASET_REPO_ID`         | `local/portal-hitl`                       | HF or local repo id                                                   |
| `DATASET_ROOT`            | (auto-discover under `../data_collection/data/`) | absolute path to a local LeRobotDataset                       |
| `RUN_NAME`                | `molmoact2-leslider`                      | output run name prefix                                                |
| `NUM_STEPS`               | `10000`                                   | total training steps                                                  |
| `BATCH_SIZE`              | `8`                                       | per-GPU batch size                                                    |
| `LR`                      | `1e-5`                                    | VLM optimizer LR                                                      |
| `ACTION_EXPERT_LR`        | `5e-5`                                    | action-expert optimizer LR                                            |
| `CHUNK_SIZE`              | `30`                                      | action horizon (matches the SO-100/101 checkpoint)                    |
| `IMAGE_KEYS`              | auto-discover                             | JSON list of `observation.images.*` keys, in policy order             |
| `CHECKPOINT_PATH`         | `allenai/MolmoAct2-SO100_101`             | starting HF checkpoint                                                |
| `MODEL_DTYPE`             | `bfloat16`                                | `bfloat16` / `float16` / `float32`                                    |
| `TRAIN_ACTION_EXPERT_ONLY`| `0`                                       | set to `1` to freeze the VLM/vision tower (fits much smaller GPUs)    |
| `ENABLE_LORA_VLM`         | `0`                                       | set to `1` to LoRA-adapt the VLM                                      |
| `SETUP_TYPE`              | `single so100/so101 robotic arm in molmoact2` | prompt-side scene description (matches the SO-100/101 norm_tag)   |
| `CONTROL_MODE`            | `absolute joint pose`                     | prompt-side action-space description                                  |
| `OUTPUT_ROOT`             | `/outputs`                                | parent dir for run outputs (the SkyPilot mount path; override locally)|

## Cloud

```bash
sky launch -c finetune-molmo skypilot.yaml \
  --env DATASET_REPO_ID=you/your-dataset \
  --env IMAGE_KEYS='["observation.images.overhead_camera","observation.images.arm_camera"]'
```

Outputs are written to the Nebius bucket mounted at `/outputs`, same as
the ACT trainer.

## What it does NOT do

  * It does not run `augment_dataset_quantile_stats.py`. We use the
    MolmoAct2 `MEAN_STD` normalization fallback that the official LeRobot
    docs recommend for datasets that ship only mean/std stats. If you
    want quantile-normalized fine-tuning, run that script offline on
    your dataset first and remove the `normalization_mapping` flag in
    `train.py`.
  * It does not handle a slider-aware fine-tune. `slider.vel` is
    stripped at the dataset boundary so the policy stays 6-DOF. The
    slider remains a human-teleop channel at inference time, same as the
    deployed `molmo_operator.py`.
  * It does not support multi-GPU directly. Wrap the `uv run python
    train.py` invocation with `accelerate launch --num_processes=N -m`
    if you need it; the MolmoAct2 LeRobot docs cover the multi-GPU path.
