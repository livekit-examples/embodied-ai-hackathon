# 2. Robot loop

The robot side does three things per tick: read hardware, publish
state and video, apply the latest action received from an operator.
Everything else (calibration, camera config, dict reshaping) is
hardware glue and lives in `utils/leslider.py` so `robot.py` stays
focused on the Portal flow.

## Boilerplate

```python
# robot.py
from livekit.portal import Action, Robot, RobotConfig

cfg = RobotConfig.from_yaml_file(CONFIG_PATH, room)
robot = Robot(cfg)
```

`Robot` is a thin facade over `Portal` with the role pinned to
`Role.ROBOT`. It exposes the publishing methods (`send_state`,
`send_video_frame`) and the receiving callbacks (`on_action`,
`on_action_chunk`). It refuses anything that only makes sense on the
operator side.

## Receiving actions

The robot does not poll. The Portal callback fires on the asyncio loop
the moment a packet arrives:

```python
latest_action: dict[str, float] = {}

def on_action(action: Action) -> None:
    nonlocal latest_action
    latest_action = {k: float(v) for k, v in action.values.items()}

robot.on_action(on_action)
```

We cache the latest values and apply them on the next tick. Two
reasons:

1. The callback runs at packet rate, which can spike above the tick
   rate during a burst. Applying every callback would over-drive the
   bus.
2. The hardware loop has its own cadence (FPS) and serial bus latency.
   Decoupling lets us stay on schedule even if a packet arrives
   mid-tick.

`action.values` is typed per the YAML schema (`bool` for `bool`,
`float` for `f32`, etc.). `action.raw_values` is the underlying f64
dict if you want to skip the per-field cast.

## Connecting

Connection is async because Portal needs an event loop bound for its
internal callbacks:

```python
async def main():
    follower.connect()                    # hardware first
    await robot.connect(url, token)       # then the wire
    ...
    try:
        async for _ in pace(fps):
            ...                           # tick body
    finally:
        await robot.disconnect()
        follower.disconnect()
```

`pace(fps)` is an async generator in `_common.py` that yields tick
indices and `await asyncio.sleep`s between them. Using a sync
`time.sleep` here would block Portal callbacks scheduled on the same
loop, so we always await.

## The tick body

```python
ts_us = int(time.time() * 1_000_000)

state, frames = split_state_frames(follower.get_observation())
robot.send_state(state, timestamp_us=ts_us)
for name, frame in frames.items():
    robot.send_video_frame(name, frame, timestamp_us=ts_us)

if latest_action:
    follower.send_action(latest_action)
```

Three points:

- **One timestamp per tick.** `send_state` and every `send_video_frame`
  share the same `ts_us`. Portal's synchronizer on the operator side
  uses these timestamps to match state samples with frames into a
  single `Observation`. Different timestamps would never match.
- **Frames are RGB uint8.** `send_video_frame` accepts `bytes` or an
  `np.ndarray(H, W, 3)`. OpenCV reads BGR, so `utils/leslider.py`
  converts before publish.
- **Anything not in the YAML schema is dropped.** We filter the
  follower's flat observation dict in `split_state_frames` to keep
  only `*.pos` and `*.vel` keys. Sending `{motor}.current` here
  without a matching YAML field would log a warning and discard.

## Things that stay out of the way

- The follower's calibration prompts run at `follower.connect()`
  before Portal connects. Any error from the bus surfaces immediately
  and we never reach the wire.
- `robot.on_active_operator_changed(...)` logs handoffs. The robot
  itself does not care who is driving; the gate has already filtered
  by the time a packet reaches `on_action`.

Next: [03. Operator loop](03-operator-loop.md).
