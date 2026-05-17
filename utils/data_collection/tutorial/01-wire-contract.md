# 1. The wire contract

Every peer in a Portal session (the robot, the human teleoperator, any
policy) needs to agree on the same wire contract: fps, video tracks,
state schema, action schema. If two peers disagree on any of these,
their packets are dropped at the receiver with a fingerprint warning.

In this project the contract lives in one YAML file at the project
root, loaded by every peer:

```yaml
# portal.yaml
version: 1
fps: 30

videos:
  - { name: arm_camera, codec: h264 }

state:
  - { name: shoulder_pan.pos,  dtype: f32 }
  - { name: shoulder_lift.pos, dtype: f32 }
  - { name: elbow_flex.pos,    dtype: f32 }
  - { name: wrist_flex.pos,    dtype: f32 }
  - { name: wrist_roll.pos,    dtype: f32 }
  - { name: gripper.pos,       dtype: f32 }
  - { name: slider.vel,        dtype: f32 }

action:
  # same fields as state in this rig
  - ...
```

## Loading per role

Three peers, three roles. Each loads the same YAML and supplies its
role at the call site:

```python
# robot.py
from livekit.portal import RobotConfig
cfg = RobotConfig.from_yaml_file("portal.yaml", room)

# teleoperator.py and policies/<algo>/inference.py
from livekit.portal import OperatorConfig
cfg = OperatorConfig.from_yaml_file("portal.yaml", room)
```

Identity (the LiveKit room name and your participant identity) is
never in the YAML. It comes from environment variables and the JWT.
That is what makes the same file reusable across robot, teleoperator,
and policy processes.

## Why YAML, not Python builders

`PortalConfig` exposes `add_state_typed(...)`, `add_video(...)`,
`add_action_chunk(...)` for building configs in code. The YAML loader
is a thin wrapper that calls the same underlying methods. We pick YAML
for two reasons:

1. The schema is shared. If the robot and the operator drift in code,
   you find out at runtime by reading a fingerprint warning. With a
   single YAML file checked into the repo, drift is impossible.
2. The schema is the dataset contract too. Keeping it in one declared
   place makes `utils/recorder.py` easy: it just reads
   `cfg.state_schema` and `cfg.action_schema` to derive feature names.

## Schema rules worth knowing

- Field order is significant. Reordering renames the fingerprint.
- Field dtypes are checked at send time. Sending an `int` for an `f32`
  field works (it widens). Sending an `int` for a `bool` field raises
  `PortalError.DtypeMismatch` before the packet leaves the sender.
- Adding a field is backwards-incompatible. All peers must update at
  once. There is no schema migration.
- The dataset writer uses the YAML schema directly. If you add motor
  currents on the wire, utils/recorder.py will pick them up the next run.

## Picking a video codec

Each `videos:` entry takes a `codec` and (for `mjpeg`) a `quality`.
The choice has real consequences for both control latency and dataset
quality:

| Codec | Path | Tradeoff |
|---|---|---|
| `h264` | WebRTC media | Lowest end-to-end latency at scale. Inter-frame prediction artifacts that degrade as training material. |
| `mjpeg` | LiveKit byte stream | Per-frame JPEG. Frame-independent, sub-ms decode. `quality: 90` is near-lossless on natural images at ~10x compression. |
| `png` | LiveKit byte stream | Lossless byte stream. ~2-3x compression. Use when bit-exact frames matter. |
| `raw` | LiveKit byte stream | Uncompressed RGB24. Largest payload; rarely justified. |

This project defaults to `mjpeg, quality: 90` because the captured
data is going into a training dataset where compression artifacts
matter. For pure inference where lowest latency wins, `h264` is the
right call. The wire schema is the same either way; only the encode
path and on-wire size differ.

## In this repo

| File | Role |
|---|---|
| `portal.yaml` | the contract |
| `robot.py` line ~25 | `RobotConfig.from_yaml_file(CONFIG_PATH, room)` |
| `teleoperator.py` line ~55 | `OperatorConfig.from_yaml_file(CONFIG_PATH, room)` |
| `policies/<algo>/inference.py` | same `OperatorConfig.from_yaml_file` |

Next: [02. Robot loop](02-robot-loop.md).
