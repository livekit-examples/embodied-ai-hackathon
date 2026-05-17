# leslider human operator

Human teleop operator for the leslider robot. Ported from
[portal-hitl](https://github.com/livekit/portal-hitl)'s `teleoperator.py`
with the dataset recorder stripped — this operator only flies the robot.

A human flies the remote leslider with a local SO-101 leader arm (plus
arrow keys for slider velocity). Observations from the robot mirror
into rerun so the operator can see the remote cameras.

## Hardware

- SO-101 leader arm on the operator machine.
- Serial port for the leader (`SO101_LEADER_PORT`).
- The robot side (`robot/`) must be running and connected to the same
  LiveKit room.

## Setup

```bash
cp .env.example .env
# Fill LIVEKIT_*, SO101_LEADER_PORT.
uv sync
```

First connect calibrates the leader and walks through ranges of
motion. Calibration files are stored under
`~/.cache/huggingface/lerobot/calibration/` keyed by `SO101_LEADER_ID`.

## Run

```bash
uv run python human_operator.py
```

The operator joins the configured LiveKit room as `human-operator`,
spawns rerun, and prints the hotkey help. Each tick, it reads one
action from the leader and sends it on the wire.

| Key | Action |
|---|---|
| `c` | Cycle the active operator (claim/release control between self and remote operators). |
| `x` | Clean quit. |

Slider drive (arrow keys) is handled by the leader's own pynput
listener: `←/→` hold to drive, `↑/↓` to trim cruise speed, space to
stop.

## Active-operator handoff

`c` cycles through `[self, *remote_operators]`. With no remote
operators (solo teleop), pressing `c` once claims control — the robot
gate drops actions whose sender isn't the active operator, so until
you press `c` the arm won't move. When a policy operator (e.g.
`molmo`) joins the same room, `c` cycles between you and the policy
for instant handoff.
