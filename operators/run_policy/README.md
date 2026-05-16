# run-policy-operator

A Portal Operator that pre-loads multiple trained ACT checkpoints and
exposes a single RPC

```
run_policy(policy: "pick_up" | "put_down" | ...)
```

to switch between them and take over the active-operator slot. The
policies are trained via `utils/data_collection/policies/act/train.py`
on data recorded with that pipeline.

## Why this exists

The shared `portal.yaml` at the repo root publishes **two** cameras
(`arm_camera` + `overhead_camera`), but ACT policies trained via
`utils/data_collection` are commonly single-camera (its local
`portal.yaml` declares only `arm_camera`). This operator inspects each
loaded checkpoint's `policy.config.input_features`, pulls only those
cameras out of the observation, and drops the rest — so the same robot
can host a two-camera VLA (Molmo) and one-camera ACT policies side by
side.

## Setup

```bash
cp ../move_to/.env.example .env   # if you don't already have one
# Fill LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_ROOM.
uv sync
```

## Run

Point the operator at one or more checkpoints. Two equivalent ways:

```bash
# Env vars (handy for two well-known policies):
PICK_UP_CHECKPOINT=/path/to/pick_up/checkpoints/050000/pretrained_model \
PUT_DOWN_CHECKPOINT=/path/to/put_down/checkpoints/050000/pretrained_model \
uv run python run_policy_operator.py

# Or repeatable --policy flags (any name=path pair):
uv run python run_policy_operator.py \
    --policy pick_up=/path/to/pick_up/.../pretrained_model \
    --policy put_down=/path/to/put_down/.../pretrained_model
```

The operator joins the configured LiveKit room as
`run-policy-operator` and parks until a peer calls
`run_policy("pick_up")` (or `"put_down"`). On call it:

1. resets the chosen policy's internal chunk / temporal ensembler,
2. claims active-operator control,
3. starts driving the robot with that policy until something else
   takes the active-operator slot.

From another operator (e.g. a controller / orchestrator):

```python
result = await op.perform_rpc(
    "run_policy",
    "pick_up",
    destination="run-policy-operator",
)
# result -> '{"policy": "pick_up",
#             "cameras": ["arm_camera"],
#             "active_operator": "run-policy-operator-..."}'
```

## Flags

```
--policy NAME=PATH                # repeat per checkpoint
--no-temporal-ensemble            # chunk-replan instead of per-tick ensemble
--temporal-ensemble-coeff 0.01    # default; lower = smoother, laggier
```

## Notes

* Both checkpoints are held in memory; only the selected one runs a
  forward pass per tick. Loading two ACT checkpoints on a single GPU
  is cheap (~hundreds of MB each); a Molmo-class VLA would be a
  different story.
* If a checkpoint declares cameras that the robot doesn't publish, the
  tick loop waits for them rather than feeding zeros. Fix this by
  retraining without that camera or by adding it to the robot side.
* State / action wire shape is the shared 7-dim leslider schema (6 arm
  `.pos` + `slider.vel`). ACT checkpoints from data_collection are
  trained on the same shape, so no shimming is needed.
