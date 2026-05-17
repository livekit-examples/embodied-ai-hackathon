# embodied-ai-hackathon

This repo is a submission to the SPC Embodied AI Hackathon. In this project, we demonstrate how a Voice AI Agent can be used to interact with a desktop robot arm on a slider.

The scope of this project includes building:

- An orchestrating voice agent to which an user can give general commands such as "Find me a tool which can help me open this box" and create code as policies to run for the robot.
- A set of policies for picking up and placing objects.
- A classical robotics policy for moving the robot on slider to various position on a desk.

Existing priors used in this project includes:

- LiveKit Agents which is a voice agents orchestration platform, our flagship product.
- LiveKit Portal which is our SDK for teleoperation and data collection on any robot. This is the backbone for orchestration of the entire project. This allows operators or policies to run from different computers in the same network and allow us to easily collect data for our more than complex system without running into configurations and hardware hassle.
- LeSlider which is a simple hardware extension to the SO101 we designed for fun and for desktop use cases like this.

## Structure

- `operators/`: operators (policies, teleop, perception) that join the same Portal room.
- `robot/`: the robot runtime (drivers, scripts, and the Portal-published robot interface).
- `utils/`: utilities for dataset collection.
- `portal.yaml`: Portal room configuration shared by the robot and operators.

## Quick Instructions

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

| flag | env | default |
| --- | --- | --- |
| `--policy NAME=PATH` | `PICK_UP_CHECKPOINT`, `PUT_DOWN_CHECKPOINT` | none (≥1 required) |
| `--reward NAME=PATH` | `PICK_UP_REWARD`, `PUT_DOWN_REWARD` | none |
| `--duration NAME=SECONDS` | `PICK_UP_DURATION`, `PUT_DOWN_DURATION` | none |
| `--reward-threshold` | `REWARD_THRESHOLD` | `0.7` |
| `--reward-trigger-ticks` | `REWARD_TRIGGER_TICKS` | `10` (~0.33 s @ 30 fps) |
| `--reward-camera` | `REWARD_CAMERA` | `arm_camera` |
| `--default-policy` | `RUN_POLICY_DEFAULT` | none (pre-warm only) |
| `--no-temporal-ensemble` / `--temporal-ensemble-coeff` | n/a | enabled / `0.01` |

A policy with neither `--reward` nor `--duration` returns immediately (`status: "no_terminator"`); the caller is on its own for completion detection.

**RPC:** `run_policy(payload)`. Payload is `"pick_up"` or `{"policy": "pick_up"}`. Reply is a JSON string:

```json
{"policy": "pick_up", "cameras": ["arm_camera"], "active_operator": "run-policy-operator", "status": "done", "reason": "reward", "reward_prob": 0.81, "ticks_above": 10}
```

`status` is one of:

| status | reason field | meaning |
| --- | --- | --- |
| `done` | `reward` | reward streak hit `--reward-trigger-ticks`. Includes `reward_prob`, `ticks_above`. |
| `done` | `timeout` | fixed duration elapsed. Includes `seconds`. |
| `no_terminator` | n/a | neither reward nor duration attached; returned immediately. |
| `preempted` | n/a | another `run_policy()` took over. Includes `by`. |
| `cancelled` | n/a | operator shut down before any terminator fired. |

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

| flag | env | default |
| --- | --- | --- |
| `--model` | `MOLMO_MODEL` | `allenai/MolmoAct2-SO100_101` |
| `--dtype` | `MOLMO_DTYPE` | `bfloat16` (`float16`, `float32` also accepted) |
| `--num-steps` | `MOLMO_NUM_STEPS` | `10` (flow-matching solver steps per chunk) |
| `--warmup` | `MOLMO_WARMUP` | `2` (pre-connect forward passes to bake the CUDA graph) |
| `--no-cuda-graph` | n/a | CUDA-graph caching on by default (auto-disabled off-CUDA) |
| `--no-normalize-language` | n/a | task-string lowercase/trim-punct on by default |
| (default task) | `LESLIDER_TASK` | `"Pick the blue cube up and hold it"` |

**RPC:** `run_molmo(payload)`. Payload is `"pick up the red cup"` or `{"prompt": "..."}`. Retargets the running policy to the new prompt, drops the in-flight chunk so the change takes effect next tick, and self-claims active-operator. Returns immediately (no completion signal). Reply is a JSON string:

```json
{"prompt": "pick up the red cup", "active_operator": "run-molmo-operator"}
```

Errors come back as `RpcError 1400` (bad JSON payload or empty prompt).

### `operators/human`

**Identity:** `human-operator`. No RPCs; control is local via hotkeys.

Teleop driver. Reads one action per tick from a physical SO-101 leader arm and forwards it to the wire. Observations stream into rerun so the operator can see the remote cameras while flying. Calibration happens on first connect and is cached under `~/.cache/huggingface/lerobot/calibration/`.

```bash
cd operators/human && uv sync
uv run python human_operator.py
```

Needs `LIVEKIT_URL`, `LIVEKIT_ROOM`, Portal creds, and `SO101_LEADER_PORT` (serial port of the leader arm). Optional: `SO101_LEADER_ID` to key the calibration cache.

**Hotkeys** (local keyboard):

| key | action |
| --- | --- |
| `c` | Cycle active-operator through `[self, *remote_operators]`. With no peers, toggles claim/release for self. |
| `x` | Clean quit. |
| `←` / `→` | Hold to drive slider velocity (handled by the leader's own listener). |
| `↑` / `↓` | Trim slider cruise speed. |
| space | Stop slider. |

Actions are streamed every tick regardless of who is active; the robot side gates by sender, so takeover via `c` is instant.
