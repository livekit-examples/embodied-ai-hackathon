# 3. Operator loop

The operator side reads a leader arm (or runs a policy) and ships one
action per tick. It also receives synchronized observations back from
the robot so it can show video, log scalars, or feed the next
inference call.

## Boilerplate

```python
# teleoperator.py
from livekit.portal import Operator, OperatorConfig

cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
op = Operator(cfg)
```

`Operator` is the role-pinned facade for `Role.OPERATOR`. It exposes
the publishing methods (`send_action`, `send_action_chunk`) and the
receiving callbacks (`on_observation`, `on_state`, `on_video_frame`,
`on_drop`).

## Receiving observations

Portal pairs each video frame with the closest matching state sample
(by timestamp) and delivers the result as a single `Observation`. The
callback hands you `state` (typed dict), `frames` (track name to
`VideoFrameData`), and `timestamp_us`:

```python
from livekit.portal import frame_bytes_to_numpy_rgb

latest_obs = None
latest_state = {}
latest_frames = {}

def on_observation(obs):
    nonlocal latest_obs
    latest_obs = obs
    latest_state.update(obs.state)
    for cam in CAMERAS:
        f = obs.frames.get(cam)
        if f is not None:
            latest_frames[cam] = frame_bytes_to_numpy_rgb(f)

op.on_observation(on_observation)
```

`frame_bytes_to_numpy_rgb` turns the wire payload (RGB24 bytes) into a
typed `(H, W, 3)` ndarray. Portal makes the same guarantee in both
directions: H264, MJPEG, PNG, RAW all decode to RGB before delivery.

## Sending actions

Send one action per tick, as a dict matching the YAML action schema:

```python
async for tick in pace(fps):
    ts_us = int(time.time() * 1_000_000)
    op.send_action(leader.get_action(), timestamp_us=ts_us)
```

`send_action` is sync and fire-and-forget. Validation runs before the
packet leaves the sender (wrong dtype raises). Whether the robot
applies it depends on the active-operator gate (see [05. Handoff](05-handoff.md)).

## Claiming control

By default the robot's `active_operator` is `None` and every action is
dropped at the gate. Someone has to set it. In this project the
teleoperator does not self-claim on connect:

```python
me = op.local_identity()
# nothing here. The human presses 'c' to claim explicitly.
```

The hotkey handler then toggles between two states: human (self) and
policy (the first remote operator):

```python
if op.active_operator() == me:
    others = op.operators()
    nxt = others[0] if others else me   # hand to policy if any
else:
    nxt = me                            # claim self
await op.set_active_operator(nxt)
```

`op.operators()` returns identities of every other operator currently
in the room (excludes self). `set_active_operator` is an RPC to the
robot; the robot updates its pointer and broadcasts the change to all
peers.

If no policy is connected, the toggle collapses to a no-op while self
is active (there is nobody else to hand to).

Policies in `policies/<algo>/inference.py` flip the default and
self-claim on connect. That makes the autonomous demo just work
without a teleoperator running, while still letting the teleop preempt
with `c` later.

## Pulling vs callbacks

Both styles work and you can mix them:

| Method | Returns |
|---|---|
| `op.on_observation(cb)` | callback fires on every synced obs |
| `op.get_observation()` | latest synced obs, or `None` if none yet |
| `op.on_video_frame(name, cb)` | callback per frame on this track |
| `op.get_video_frame(name)` | latest frame on this track |

The synchronized callback path has lower latency (no polling delay)
and is what the policies use. The pull path is convenient for one-off
inspection or for tightly synchronous code.

## Lifecycle

Same shape as the robot side:

```python
async def main():
    leader.connect()
    await op.connect(url, token)
    ...
    try:
        async for tick in pace(fps):
            ...
    finally:
        await op.disconnect()
        leader.disconnect()
```

Next: [04. HITL recording](04-hitl-recording.md).
