# move_to_debug operator

Read-only sibling of [`move_to`](../move_to) that exercises the same
perception stack — Moondream Cloud's `point` endpoint plus the
AprilTag detector on the overhead camera — but **never claims active
operator control and never publishes actions**.

Use it to verify the perception layer (camera → tag detector,
camera → Moondream) by comparing the slider tag's pixel-Y against the
target object's pixel-Y — the same axis the real `move_to_operator`
drives — without putting the slider in motion.

## Run

```bash
cp .env.example .env  # fill in LIVEKIT_* and MOONDREAM_API_KEY
uv sync
uv run move_to_debug_operator.py
```

You also need a robot process (e.g. `robot.py` from
[portal-hitl](https://github.com/livekit-examples/portal-hitl)) joined
to the same `LIVEKIT_ROOM`, otherwise no camera frames ever arrive and
every `move_to` call returns `1404`.

## Wire contract

```text
move_to(payload) -> JSON
```

Same payload shape as `move_to_operator`:

- `payload` is either a bare string (`"blue cube"`) or a JSON object
  (`{"description": "blue cube"}`).

The RPC blocks for up to `DEBUG_TRACK_TIMEOUT_S` seconds (default
20s). While it runs:

- On every Portal tick (~30 Hz), the AprilTag detector runs on the
  latest `overhead_camera` frame and the slider tag's pixel-Y is
  printed to console.
- Every `DEBUG_OBJECT_DETECT_INTERVAL_S` seconds (default 1s),
  Moondream's `point` endpoint is called again to relocate the
  requested object and its pixel-Y is printed.

It returns a JSON summary like:

```json
{
  "description": "blue cube",
  "frame": {"width": 640, "height": 480},
  "initial_target": {"y": 0.33, "y_px": 156.8},
  "last_object":   {"y": 0.33, "y_px": 156.8},
  "last_tag":      {"y_px": 187.2},
  "tag_iters":      600, "tag_hits":   598, "tag_miss":  2,
  "object_iters":    20, "object_hits":  20, "object_miss": 0,
  "elapsed_s":  20.01,
  "took_control": false
}
```

Errors:

- `1400` — empty or malformed payload.
- `1404` — no frame yet, or the object was not found in the initial
  frame.
- `1502` — Moondream API error on the initial detection.

## Call it

```bash
# in another shell, with the operator already running
uv run test_move_to_debug.py --object "blue cube"
```

Or address it from any peer in the same room:

```python
result = await op.perform_rpc(
    "move_to", "blue cube", destination="move-to-debug-operator"
)
```

The destination identity is what differs from `move_to_operator`; the
RPC method name (`move_to`) and payload shape are identical, so the
existing `test_move_to.py` works against this operator with
`--destination move-to-debug-operator`.

## Knobs

See `.env.example` for the complete list. The most useful ones:

- `DEBUG_TRACK_TIMEOUT_S` — how long the loop runs (default 20s).
- `DEBUG_TAG_PRINT_EVERY_TICKS` — throttle tag console output
  (default 1 = every tick).
- `DEBUG_OBJECT_DETECT_INTERVAL_S` — Moondream re-detection cadence
  (default 1s).
- `SLIDER_APRILTAG_FAMILY` / `SLIDER_APRILTAG_ID` — same meaning as in
  `move_to_operator`; pin to the slider's tag if other tags are in
  the frame.
