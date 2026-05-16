# move_to operator

Perception-only operator that answers `move_to(description)` RPCs by
running Moondream Cloud's `point` endpoint against the latest
`overhead_camera` frame seen on the Portal room.

This is a **sensor**, not a controller — it returns the normalized
centre of the described object and nothing else. The actual slider
control loop lives elsewhere (a follow-up operator can subscribe to
this one's output and convert `(x, y)` into a `slider.vel` command).

## Run

```bash
cp .env.example .env  # fill in LIVEKIT_* and MOONDREAM_API_KEY
uv run move_to_operator.py
```

You also need a robot process (e.g. `robot.py` from
[portal-hitl](https://github.com/livekit-examples/portal-hitl)) joined
to the same `LIVEKIT_ROOM`, otherwise no camera frames ever arrive and
every `move_to` call returns a `no 'overhead_camera' frame yet` error.

## Wire contract

```text
move_to(payload) -> JSON
```

- `payload` is either a bare string (`"blue cube"`) or a JSON object
  (`{"description": "blue cube"}`).
- Returns `{"x": <0..1>, "y": <0..1>, "width": <px>, "height": <px>}`
  where `x`/`y` is Moondream's normalized centre of the first detected
  instance, and `width`/`height` is the camera frame size in pixels.
- Errors:
  - `1400` — empty or malformed payload.
  - `1404` — no frame yet, or the object was not found.
  - `1502` — Moondream API error.

## Call it

From any peer (operator, agent, script) in the same room:

```python
result = await op.perform_rpc(
    "move_to",
    "blue cube",
    destination="move_to_operator",
)
# result -> '{"x": 0.43, "y": 0.61, "width": 640, "height": 480}'
```
