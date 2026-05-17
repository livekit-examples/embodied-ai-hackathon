# portal-hitl

An end-to-end example of human-in-the-loop teleop and data recording
for a [leslider](https://github.com/pham-tuan-binh/leslider) (SO-101
arm on a linear slider) over
[`livekit-portal`](https://github.com/livekit/livekit-portal).

A human flies the remote leslider with a local SO-101 leader (plus
arrow keys for slider velocity). At any moment they can hand control
off to a trained policy (ACT or Diffusion). The teleoperator process
also records: every executed action, human or policy, is paired with
the synchronized observation and written to a `LeRobotDataset` episode.

## Layout

```
portal-hitl/
├── portal.yaml             # shared wire contract
├── _common.py              # env loader, token minter, async pacer
├── robot.py                # Portal Robot flow, drives the leslider
├── teleoperator.py         # Portal Operator flow, leader + recorder
├── utils/                  # hardware builders, rerun blueprint, hotkeys, recorder
├── policies/               # ACT + Diffusion: inference.py / train.py / skypilot.yaml
├── scripts/                # deploy_to_robot.sh, deploy.rsyncignore
└── tutorial/               # walkthrough of every Portal pattern used here
```

## Setup

You need:

- A LiveKit project (URL, API key, API secret) for the wire layer.
- A leslider rig: SO-101 follower arm mounted on a linear slider, one
  arm-mounted USB camera.
- A separate SO-101 leader arm for teleop, on whichever machine runs
  `teleoperator.py`.
- Python 3.12+, `uv`.

```bash
git clone <this repo>
cd portal-hitl
cp .env.example .env       # fill in LIVEKIT_*, serial ports, camera devices
uv sync
```

Requires `livekit-portal>=0.2.1` (the YAML loader and
`set_action_subscription` both landed post-0.2.0).

### Calibrate the arms

The lerobot follower and leader each need calibration before first
use. Both sides run the standard lerobot prompts on first connect:
center the arm, walk through ranges of motion, save. Calibration
files are stored under `~/.cache/huggingface/lerobot/calibration/`
keyed by the `LESLIDER_ID` and `SO101_LEADER_ID` env vars. Once
calibrated, subsequent connects accept the saved file.

If `robot.py` or `teleoperator.py` blocks at startup waiting for
input, that's the calibration prompt; press ENTER to use the saved
file or `c` then ENTER to recalibrate.

### Find the right serial ports / cameras

Linux: `ls /dev/serial/by-id/` for the arms, `v4l2-ctl --list-devices`
for cameras (use `/dev/video*`). macOS: `ls /dev/tty.usbmodem*`,
camera index is an integer (0, 1, ...) per AVFoundation.

## Collect data

Three terminals, three processes. Robot and teleop must connect to
the same LiveKit room:

```bash
# Terminal 1: physical leslider host.
uv run robot.py
# Waits for camera frames + serial bus, prints "connected" when ready.

# Terminal 2: human operator's machine, with the SO-101 leader.
uv run teleoperator.py
# Spawns rerun, prints hotkey help.
```

Once both are connected, on the teleoperator window:

| Key | Action |
|---|---|
| `c` | Toggle active operator between human (self) and policy. The arm only moves when an operator is active. |
| `r` | Toggle episode recording. |
| `[` | Discard the in-flight episode without saving. |
| `x` | Clean quit. |

Slider drive lives on the leader's own listener: `←/→` hold to drive,
`↑/↓` to trim cruise speed, space to stop.

A typical recording session:

```
(robot.py and teleoperator.py both running, no operator active)
press c        → arm follows the leader
move arm to start position
press r        → episode recording starts
perform task
press r        → episode ends, saves in background
... repeat ...
press x        → quit, finalizes any in-flight save
```

Episodes land in `data/<repo_id>/` (default `data/local/portal-hitl/`).
Override with `PORTAL_HITL_DATASET_REPO_ID` and `PORTAL_HITL_DATASET_ROOT`.
The recorder resumes if a dataset already exists, so a long-running
corpus can be collected across multiple sessions.

## Train a policy

`policies/<algo>/train.py` wraps `lerobot.scripts.lerobot_train` with
the leslider-specific shims (DataLoader tuning, FUSE-safe symlinks).
Configure via env:

```bash
DATASET_REPO_ID=you/your-recording-repo \
NUM_STEPS=20000 \
BATCH_SIZE=64 \
uv run --only-group train python policies/act/train.py
```

For cloud GPU runs, each algo ships a `skypilot.yaml`:

```bash
sky launch policies/act/skypilot.yaml -e DATASET_REPO_ID=you/your-repo
```

Checkpoints are saved every quarter of total steps to
`/outputs/<RUN_NAME>/checkpoints/<NNNNNN>/pretrained_model/`.

## Run inference

Hand control off to a trained checkpoint with the same robot and
teleop running in the background:

```bash
# Terminal 3: any machine with the LiveKit creds and the checkpoint.
uv run policies/act/inference.py --checkpoint path/to/025000

# or for Diffusion:
uv run policies/diffusion/inference.py --checkpoint path/to/025000 \
    --async-predict --num-inference-steps 20
```

The policy starts disengaged. Press SPACE in the policy window to
engage; press again to disengage. Press `x` to quit. By default the
policy self-claims the active operator on connect, so it can drive
even without a teleop running. With a teleop also running, the
teleop's `c` hotkey takes control back from the policy mid-episode
(useful for HITL corrections, all of which are recorded).

Useful flags:

```
policies/act/inference.py
  --checkpoint PATH                # or env ACT_CHECKPOINT
  --no-temporal-ensemble           # execute full chunk before replanning
  --temporal-ensemble-coeff 0.01   # smoother (lower) vs more reactive (higher)
  --no-claim                       # don't auto-claim active operator

policies/diffusion/inference.py
  --checkpoint PATH                # or env DIFFUSION_CHECKPOINT
  --num-inference-steps 20         # fewer denoising steps, faster, lower fidelity
  --scheduler DDIM                 # override scheduler at inference
  --async-predict                  # overlap forward pass with chunk dispatch
  --blend                          # cross-fade between chunks (needs --async-predict)
  --anchor-prefix 4                # constrain chunk start (needs --async-predict)
  --no-claim
```

## Deploy to the robot host

When the leslider host is a separate machine (the common case),
`scripts/deploy_to_robot.sh` rsyncs the repo over SSH and prints
follow-up commands.

```bash
# One-off, with the remote on the command line.
REMOTE=robotuser@robot.local ./scripts/deploy_to_robot.sh

# Or set defaults in .env so plain `./scripts/deploy_to_robot.sh` works:
#   PORTAL_HITL_ROBOT_REMOTE=robotuser@robot.local
#   PORTAL_HITL_ROBOT_REMOTE_ROOT=~/workspace      # default ~/workspace
```

What it does:

1. `mkdir -p` the destination on the remote.
2. `rsync -azP --delete` the project tree, honoring
   `scripts/deploy.rsyncignore`.
3. Print the next steps to run on the robot.

What gets synced and what doesn't:

| Synced | Excluded |
|---|---|
| Source code, `portal.yaml`, `pyproject.toml`, `uv.lock` | `.git`, `.venv`, Python caches |
| `.env` (so the robot inherits LIVEKIT and serial-port config) | `.env.local` (each machine keeps its own overrides) |
| `policies/` source | `data/`, `checkpoints/`, `outputs/` (operator/GPU side only) |

After the rsync, on the robot:

```bash
cd ~/workspace/portal-hitl
uv sync
uv run robot.py
```

First-time calibration walks through ranges of motion when `robot.py`
connects with no saved file; saved files live under
`~/.cache/huggingface/lerobot/calibration/` keyed by `LESLIDER_ID`.

If you change anything locally, just rerun the script. `--delete`
keeps the remote tree byte-identical to your working copy minus the
exclusions.

## Where to put each process

The three Portal peers (robot, teleop, policy) can run on the same
machine or different ones. Common splits:

| | robot.py | teleoperator.py | policies/<algo>/inference.py |
|---|---|---|---|
| Single machine demo | localhost | localhost | localhost |
| Remote teleop, local policy | leslider host | operator desk | operator desk |
| Cloud policy | leslider host | operator desk | cloud GPU box |

LiveKit handles the wire in all cases. The `portal.yaml` does not
hardcode IPs; only `LIVEKIT_URL` does, and that's an env var.

## Learn

The `tutorial/` folder walks through the Portal patterns this code
uses, in order:

1. [Wire contract](tutorial/01-wire-contract.md). `portal.yaml`,
   `RobotConfig` / `OperatorConfig`, schema fingerprinting, codec
   choice.
2. [Robot loop](tutorial/02-robot-loop.md). `send_state`,
   `send_video_frame`, `on_action`.
3. [Operator loop](tutorial/03-operator-loop.md). `send_action`,
   `on_observation`, claiming control.
4. [HITL recording](tutorial/04-hitl-recording.md).
   `action_subscription`, `reuse_stale_frames`, on_action-driven
   recording, alignment, recording while a policy drives.
5. [Handoff](tutorial/05-handoff.md). The active-operator gate and
   toggling between human and policy.
6. [Plugging in a policy](tutorial/06-plugging-in-policies.md). How
   `policies/<algo>/inference.py` calls `select_action` per tick.
