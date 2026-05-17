# Team LiveKit: Desktop Robot Assistant

We built a system of robots for assisting you on your desktop. You can do tasks such as desk cleaning or organization just by talking to a voice agent.

For example, you can tell the robot to "put the screwdriver away" and it would pick up the screwdriver on one end of the table and put it to the bin at the other end of the table.

Behind the scene this is the work of multiple systems working together.

First, the robots are orchestrated by a voice agent running on LiveKit. This agent has access to the overview of the table and can use such to make plans based on user-given commands.

If the user give a task like clean the table, the agent would first check what objects are on the table and utilize the tools it have to execute the goal.

The agent has access to 3 tools.

The first tool is `move_to`, which is a PID loop for controlling the robot slider to any object on the table. The robot is localized using an April tag while the object is localized using a VLM.

The second tool is `run_policy`, which allows us to run any of our trained ACT policy on demand.

We collected data and trained 2 policy at SPC:

- Pick up is an ACT policy trained on 200 episodes of picking random stuff up at SPC. It is used when the agent wants to pick stuff on the table. We also trained a sparse reward model so the agent knows when the pick up policy has finished and can stop it.
- Put down is another policy trained on 50 episodes of putting down stuff. However, it didn't work well so we just sample trajectories from the dataset randomly.

The third tool is `run_molmo`, which utilizes MolmoACT2, a generalized VLA that can do anything given. We fine-tuned this on all the dataset we gathered at SPC so the model can better adapt to our embodiment. Results show that the model can reliably pick up and localize objects that was not in trained dataset, showing its capabilities to generalize.

Behind the scene, the entire system is powered by our own arbitration and network infrastructure. The robot is not controlled from a single computer.

The voice agent lives on one laptop. The ACT policies live on one laptop. The slider control lives on one laptop. MolmoACT2 lives on a H200 instance in Finland.

Through this project, we showcase how robotics is a systems problem, how humans can interact with robots, how we can deploy robots in the wild with LiveKit and how the future will look like.

## Datasets

All datasets are `LeRobotDataset` v3.0 captures collected at SPC on the leslider rig (SO-101 follower on a linear slider). `binhpham/spc-pick-stuff` is the merged corpus used to train the `pick_up` ACT policy and the sparse-reward classifier.

| Dataset                                                                                          | Object      | Task | Episodes | Prompt                                | Visualize                                                                                                                   |
| ------------------------------------------------------------------------------------------------ | ----------- | ---- | -------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [`binhpham/spc-pick-granola-bar`](https://huggingface.co/datasets/binhpham/spc-pick-granola-bar) | granola     | pick | 50       | "pick up the granola bar and hold it" | [view](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fbinhpham%2Fspc-pick-granola-bar%2Fepisode_0%3Ft%3D9) |
| [`binhpham/spc-pick-sharpie`](https://huggingface.co/datasets/binhpham/spc-pick-sharpie)         | sharpie     | pick | 50       | "pick up the sharpie and hold it"     | [view](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fbinhpham%2Fspc-pick-sharpie%2Fepisode_0)             |
| [`binhpham/spc-pick-screwdriver`](https://huggingface.co/datasets/binhpham/spc-pick-screwdriver) | screwdriver | pick | 50       | "pick up the screwdriver and hold it" | [view](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fbinhpham%2Fspc-pick-screwdriver%2Fepisode_0)         |
| [`binhpham/spc-pick-stuff`](https://huggingface.co/datasets/binhpham/spc-pick-stuff)             | all         | pick | 200      |                                       | [view](https://huggingface.co/spaces/lerobot/visualize_dataset)                                                             |
| [`binhpham/spc-mock-put-down`](https://huggingface.co/datasets/binhpham/spc-mock-put-down)       | nothing     | put  | 50       | "mock put down"                       | [view](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fbinhpham%2Fspc-mock-put-down%2Fepisode_0)            |
| [`j1823/spc-pick-up-scissors`](https://huggingface.co/datasets/j1823/spc-pick-up-scissors)       | scissors    | pick | 50       | "pick up scissors"                    | [view](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fj1823%2Fspc-pick-up-scissors%2Fepisode_0)            |

## Models

| Model                                                                                     | What                                                                         | Trained on                                          |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------- |
| [`binhpham/spc-pick-stuff-act`](https://huggingface.co/binhpham/spc-pick-stuff-act)        | ACT policy that drives `run_policy pick_up`.                                 | `binhpham/spc-pick-stuff` (the merged pick corpus). |
| [`binhpham/spc-pick-stuff-reward`](https://huggingface.co/binhpham/spc-pick-stuff-reward) | Sparse-reward classifier that signals when the `pick_up` ACT policy is done. | `binhpham/spc-pick-stuff`.                          |
| [`binhpham/molmoact2-leslider`](https://huggingface.co/binhpham/molmoact2-leslider)       | MolmoAct2 fine-tuned for the leslider rig. Powers `run_molmo`.               | All SPC pick captures.                              |

## Priors

We built the below components before the hackathon:

- **[LiveKit Agents](https://github.com/livekit/agents)**: our flagship voice agent orchestration platform.
- **[LiveKit Portal](https://github.com/livekit/livekit-portal)**: our SDK for teleoperation and data collection on any robot. The backbone of this project. It lets operators and policies run on different computers in the same network without running into configuration or hardware hassle, and it makes data collection on a complex multi-machine system painless.
- **[LeSlider](https://github.com/pham-tuan-binh/leslider)**: a simple hardware extension to the SO-101 that we designed for fun and for desktop use cases like this.

## Structure

- `operators/`: operators (policies, teleop, perception) that join the same Portal room.
- `robot/`: the robot runtime (drivers, scripts, and the Portal-published robot interface).
- `orchestrator/`: the voice agent that interprets user commands and dispatches operator RPCs.
- `ui/`: the web UI for talking to the orchestrator and watching the robot.
- `utils/`: dataset collection, dataset merging, and policy/reward training scripts.
- `portal.yaml`: Portal room configuration shared by the robot and operators.

## Robot

`robot/robot.py` is the robot-side runtime. It connects to the physical leslider SO-101 follower and both cameras, joins the configured LiveKit room as `robot`, publishes state and video frames, and applies the latest incoming Portal action. If the active operator disconnects, the robot clears the active-operator slot so the arm does not keep holding a stale command.

```bash
cd robot && cp .env.example .env  # fill in LIVEKIT_* and LESLIDER_*
uv sync
uv run robot.py
```

`portal.yaml` defines the wire schema (two MJPEG camera streams, seven state fields, seven matching action fields). Hardware knobs live in env: `LESLIDER_PORT`, `LESLIDER_ID`, `LESLIDER_CAM_ARM`, `LESLIDER_CAM_OVERHEAD`, `LESLIDER_CAM_WIDTH` / `LESLIDER_CAM_HEIGHT`. First run may prompt for LeRobot calibration; calibration files are cached under `~/.cache/huggingface/lerobot/calibration/` keyed by `LESLIDER_ID`.

## Orchestrator

`orchestrator/src/agent.py` is a LiveKit Agents voice agent. It joins the same room as the robot, subscribes to the `overhead_camera` track for visual context, and exposes a small toolset that dispatches operator RPCs:

- `plan_and_execute`: vision-grounded planner LLM that turns a free-form request ("get me the wrench") into a sequence of `move_to` + `run_policy` steps, then narrates progress aloud as each step runs.
- `run_molmo`: gated on the user explicitly naming MolmoAct. Forwards a prompt to `run-molmo-operator`.
- `reset_pose` / `release_gripper`: thin wrappers over the matching `move-to-operator` RPCs, triggered when the user asks the arm to park, go home, or let go.

```bash
cd orchestrator && uv sync
uv run python src/agent.py download-files  # one-off: pulls Silero VAD + turn-detector
uv run python src/agent.py dev             # use `console` for terminal, `start` for prod
```

Needs `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in `.env.local`. The pipeline uses LiveKit Inference for STT (Deepgram Nova-3), TTS (Cartesia Sonic-3), and the planner / voice LLMs (OpenAI GPT). A Dockerfile is included for LiveKit Cloud deployment.

## UI

`ui/` is a Next.js web frontend (LiveKit Agents starter for React) that the user talks to. It mints a LiveKit token via `app/api/`, joins the same room as the orchestrator and robot, and renders the voice control bar, chat transcript, and remote media tiles. Branding, theme, and copy are configured in `app-config.ts`.

```bash
cd ui && pnpm install
pnpm dev
```

Needs `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in `.env.local` so the token route can sign joins. The dev server runs at `http://localhost:3000`.

## Operators

### `operators/run_policy`

**Identity:** `run-policy-operator`. Target this when calling the RPC.

Drives the robot with pre-loaded ACT checkpoints. The RPC blocks until a "task done" signal fires (sparse-reward classifier and/or fixed duration), then parks. `pick_up` typically uses the reward classifier; `put_down` uses a fixed duration.

```bash
cd operators/run_policy && uv sync
uv run python run_policy_operator.py \
  --policy pick_up=/path/to/pick_up_ckpt \
  --policy put_down=/path/to/put_down_ckpt \
  --reward pick_up=/path/to/reward/best.pt \
  --duration put_down=4.0
```

Needs `LIVEKIT_URL`, `LIVEKIT_ROOM`, and Portal creds in env / `.env`.

**Flags** (all env-overridable):

| flag                                                   | env                                         | default                 |
| ------------------------------------------------------ | ------------------------------------------- | ----------------------- |
| `--policy NAME=PATH`                                   | `PICK_UP_CHECKPOINT`, `PUT_DOWN_CHECKPOINT` | none (≥1 required)      |
| `--reward NAME=PATH`                                   | `PICK_UP_REWARD`, `PUT_DOWN_REWARD`         | none                    |
| `--duration NAME=SECONDS`                              | `PICK_UP_DURATION`, `PUT_DOWN_DURATION`     | none                    |
| `--reward-threshold`                                   | `REWARD_THRESHOLD`                          | `0.7`                   |
| `--reward-trigger-ticks`                               | `REWARD_TRIGGER_TICKS`                      | `10` (~0.33 s @ 30 fps) |
| `--reward-camera`                                      | `REWARD_CAMERA`                             | `arm_camera`            |
| `--default-policy`                                     | `RUN_POLICY_DEFAULT`                        | none (pre-warm only)    |
| `--no-temporal-ensemble` / `--temporal-ensemble-coeff` | n/a                                         | enabled / `0.01`        |

A policy with neither `--reward` nor `--duration` returns immediately (`status: "no_terminator"`); the caller is on its own for completion detection.

**RPC:** `run_policy(payload)`. Payload is `"pick_up"` or `{"policy": "pick_up"}`. Reply is a JSON string:

```json
{
  "policy": "pick_up",
  "cameras": ["arm_camera"],
  "active_operator": "run-policy-operator",
  "status": "done",
  "reason": "reward",
  "reward_prob": 0.81,
  "ticks_above": 10
}
```

`status` is one of:

| status          | reason field | meaning                                                                            |
| --------------- | ------------ | ---------------------------------------------------------------------------------- |
| `done`          | `reward`     | reward streak hit `--reward-trigger-ticks`. Includes `reward_prob`, `ticks_above`. |
| `done`          | `timeout`    | fixed duration elapsed. Includes `seconds`.                                        |
| `no_terminator` | n/a          | neither reward nor duration attached; returned immediately.                        |
| `preempted`     | n/a          | another `run_policy()` took over. Includes `by`.                                   |
| `cancelled`     | n/a          | operator shut down before any terminator fired.                                    |

Errors come back as `RpcError`: `1400` (bad payload) or `1404` (unknown policy).

### `operators/molmo`

**Identity:** `run-molmo-operator`. Target this when calling the RPC.

Runs `allenai/MolmoAct2-SO100_101` (or any compatible MolmoAct2 checkpoint) as a continuous policy. Forward passes produce 30-step action chunks (~1 s at 30 fps); the wrapper appends `slider.vel = 0.0` so the slider parks while the arm runs autonomously. Best on a CUDA GPU (~16 GB VRAM in bfloat16); CPU/MPS work but are slow.

```bash
cd operators/molmo && uv sync
LESLIDER_TASK="Pick the blue cube up and hold it" \
uv run python molmo_operator.py
```

Needs `LIVEKIT_URL`, `LIVEKIT_ROOM`, and Portal creds in env / `.env`.

**Flags** (all env-overridable):

| flag                      | env               | default                                                   |
| ------------------------- | ----------------- | --------------------------------------------------------- |
| `--model`                 | `MOLMO_MODEL`     | `allenai/MolmoAct2-SO100_101`                             |
| `--dtype`                 | `MOLMO_DTYPE`     | `bfloat16` (`float16`, `float32` also accepted)           |
| `--num-steps`             | `MOLMO_NUM_STEPS` | `10` (flow-matching solver steps per chunk)               |
| `--warmup`                | `MOLMO_WARMUP`    | `2` (pre-connect forward passes to bake the CUDA graph)   |
| `--no-cuda-graph`         | n/a               | CUDA-graph caching on by default (auto-disabled off-CUDA) |
| `--no-normalize-language` | n/a               | task-string lowercase/trim-punct on by default            |
| (default task)            | `LESLIDER_TASK`   | `"Pick the blue cube up and hold it"`                     |

**RPC:** `run_molmo(payload)`. Payload is `"pick up the red cup"` or `{"prompt": "..."}`. Retargets the running policy to the new prompt, drops the in-flight chunk so the change takes effect next tick, and self-claims active-operator. Returns immediately (no completion signal). Reply is a JSON string:

```json
{ "prompt": "pick up the red cup", "active_operator": "run-molmo-operator" }
```

Errors come back as `RpcError 1400` (bad JSON payload or empty prompt).

### `operators/move_to`

**Identity:** `move-to-operator`. Target this when calling the RPCs.

Perception + classical control. Uses Moondream Cloud's `point` endpoint on the latest `overhead_camera` frame to locate the described object, then runs a bang-bang slider loop that drives an AprilTag (mounted on the slider) toward the target's pixel-X. Also exposes `reset_pose` (ramp arm joints back to a safe rest pose) and `release_gripper` (ramp gripper open while holding the arm steady). Each RPC self-claims active-operator, holds `slider.vel = 0` on the arm-only moves, and releases on exit.

```bash
cd operators/move_to && uv sync
uv run python move_to_operator.py
```

Needs `LIVEKIT_URL`, `LIVEKIT_ROOM`, Portal creds, and `MOONDREAM_API_KEY` in env / `.env`.

**Knobs** (env-only):

| env                                    | default   | purpose                                                                                      |
| -------------------------------------- | --------- | -------------------------------------------------------------------------------------------- |
| `SLIDER_APRILTAG_FAMILY`               | `tag16h5` | AprilTag family on the slider fiducial.                                                      |
| `SLIDER_APRILTAG_ID`                   | none      | optional integer; if set, only that tag ID is accepted.                                      |
| `SLIDER_MOVE_VELOCITY`                 | `6000`    | bang-bang slider speed (raw motor units).                                                    |
| `SLIDER_MOVE_THRESHOLD_PX`             | `20`      | pixel tolerance on X.                                                                        |
| `SLIDER_MOVE_TIMEOUT_S`                | `20`      | move loop timeout in seconds.                                                                |
| `SLIDER_TAG_MISS_LIMIT`                | `30`      | consecutive missing-tag ticks before bailing.                                                |
| `SLIDER_REACHED_HOLD_TICKS`            | `3`       | in-window ticks required before declaring `reached`.                                         |
| `SLIDER_VEL_INVERT`                    | `false`   | flip the sign of `slider.vel` if the rig drives the wrong way.                               |
| `SLIDER_TAG_OFFSET_X_PX`               | `0`       | flat pixel offset from tag center to gripper center along X.                                 |
| `SLIDER_TAG_OFFSET_{NEAR,FAR}_{X_,}PX` | none      | optional two-point perspective calibration; overrides the flat offset when all four are set. |
| `RESET_POSE_*` (per joint)             | rig-tuned | per-joint target angles for `reset_pose`.                                                    |
| `RESET_POSE_MAX_STEP`                  | `2.0`     | per-tick command delta during the ramp.                                                      |
| `RESET_POSE_TOLERANCE`                 | `5.0`     | per-joint convergence tolerance (degrees).                                                   |
| `RESET_POSE_TIMEOUT_S`                 | `10`      | ramp loop timeout in seconds.                                                                |

**RPCs:**

- `move_to(payload)`: payload is `"blue cube"` or `{"description": "blue cube"}`. Blocks until the slider is centered on the object, the tag goes missing, the slider hits its safe pixel range, or the timeout elapses.
- `reset_pose()`: ignores payload. Ramps all 6 arm joints to the configured rest pose with `slider.vel = 0`.
- `release_gripper()`: ignores payload. Ramps `gripper.pos` open while mirroring the rest of the arm.

Reply is a JSON string. `move_to` returns:

```json
{
  "description": "blue cube",
  "target": { "x": 0.43, "y": 0.61, "width": 640, "height": 480 },
  "reached": true,
  "reason": "reached",
  "iterations": 47,
  "elapsed_s": 1.57,
  "final_tag_x_px": 291.2
}
```

`move_to` `reason` values: `reached`, `timeout`, `tag_lost`, `at_limit`. `reset_pose` / `release_gripper` return `target` / `final` joint dicts plus `reached`, `reason` (`reached` or `timeout`), `iterations`, `elapsed_s`.

Errors come back as `RpcError`: `1400` (bad payload), `1404` (no overhead frame yet, or object not found), `1409` (no robot state yet), `1502` (Moondream API error).

### `operators/human`

**Identity:** `human-operator`. No RPCs; control is local via hotkeys.

Teleop driver. Reads one action per tick from a physical SO-101 leader arm and forwards it to the wire. Observations stream into rerun so the operator can see the remote cameras while flying. Calibration happens on first connect and is cached under `~/.cache/huggingface/lerobot/calibration/`.

```bash
cd operators/human && uv sync
uv run python human_operator.py
```

Needs `LIVEKIT_URL`, `LIVEKIT_ROOM`, Portal creds, and `SO101_LEADER_PORT` (serial port of the leader arm). Optional: `SO101_LEADER_ID` to key the calibration cache.

**Hotkeys** (local keyboard):

| key       | action                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------- |
| `c`       | Cycle active-operator through `[self, *remote_operators]`. With no peers, toggles claim/release for self. |
| `x`       | Clean quit.                                                                                               |
| `←` / `→` | Hold to drive slider velocity (handled by the leader's own listener).                                     |
| `↑` / `↓` | Trim slider cruise speed.                                                                                 |
| space     | Stop slider.                                                                                              |

Actions are streamed every tick regardless of who is active; the robot side gates by sender, so takeover via `c` is instant.

## Utils

Four standalone `uv` projects that produce the assets the operators above need: episodes for training, merged corpora, and trained ACT / reward / MolmoAct2 checkpoints. Each has its own `pyproject.toml` and `README.md` with the full knob list; the summaries below cover the happy path.

### `utils/data_collection`

End-to-end teleop + recorder. A human flies the leslider with a local SO-101 leader and the recorder writes every executed action paired with its synchronized observation into a `LeRobotDataset` under `data/<repo_id>/`. Three terminals: `robot.py` on the leslider host, `teleoperator.py` on the operator desk, optionally an `inference.py` for an ACT or Diffusion policy that can take over via the `c` hotkey.

```bash
cd utils/data_collection && cp .env.example .env && uv sync
uv run robot.py            # terminal 1, leslider host
uv run teleoperator.py     # terminal 2, operator desk (SO-101 leader)
```

Hotkeys on the teleop window: `c` toggle active operator, `r` toggle recording, `[` discard in-flight episode, `x` quit. Override the output repo with `PORTAL_HITL_DATASET_REPO_ID` / `PORTAL_HITL_DATASET_ROOT`. The same folder also ships ACT and Diffusion trainers under `policies/<algo>/train.py` (env-driven, optional SkyPilot YAMLs) and a `scripts/deploy_to_robot.sh` rsync helper. See `utils/data_collection/tutorial/` for a walkthrough of the Portal patterns used here.

### `utils/merge_datasets`

Concatenate multiple `LeRobotDataset` v3.0 directories produced by `data_collection` into a single training corpus, without re-encoding videos. Inputs must agree on `fps`, `robot_type`, feature dtypes/shapes, and video keys.

```bash
cd utils/merge_datasets && uv sync
uv run python merge_datasets.py \
    /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c \
    --output /path/to/merged --repo-id you/your-merged-repo
```

Pass `--overwrite` to clobber the output dir. Also importable as `from merge_datasets import merge_datasets`.

### `utils/train_sparse_reward`

Trains the sparse-reward classifier consumed by `operators/run_policy` (`--reward NAME=PATH`). Single-step inputs (one `arm_camera` frame + 7-d proprio), label is `1` for the last 15 frames of each episode (~0.5 s @ 30 fps), `0` elsewhere. ResNet18 + small MLP head.

```bash
cd utils/train_sparse_reward && uv sync
DATASET_REPO_ID=you/your-dataset \
DATASET_ROOT=../data_collection/data \
uv run python train.py
```

Checkpoints land in `outputs/<RUN_NAME>/`; `best.pt` is the one to point `--reward` at. Inference helper:

```python
from inference import RewardScorer
scorer = RewardScorer.from_checkpoint("outputs/<run>/best.pt")
prob = scorer.score(rgb_uint8_hxwx3, state_7_floats)
```

### `utils/finetune_molmo`

Fine-tunes `allenai/MolmoAct2-SO100_101` on a `data_collection`-style dataset for use with `operators/molmo`. The trainer slices `slider.vel` out at the dataset boundary so the policy stays 6-DOF (the slider remains a human-teleop channel at inference time). Needs a CUDA box; full fine-tune is ~48 GiB at `batch_size=8`, action-expert-only fits a single H100 at ~16 GiB.

```bash
cd utils/finetune_molmo && uv sync
DATASET_REPO_ID=you/your-dataset \
DATASET_ROOT=$(pwd)/../data_collection/data/you/your-dataset \
OUTPUT_ROOT=$HOME/outputs \
uv run python train.py
```

Cloud-GPU shortcut: `sky launch -c finetune-molmo skypilot.yaml --env DATASET_REPO_ID=...`. Set `TRAIN_ACTION_EXPERT_ONLY=1` to freeze the VLM/vision tower for smaller GPUs; set `ENABLE_LORA_VLM=1` to LoRA-adapt the VLM.
