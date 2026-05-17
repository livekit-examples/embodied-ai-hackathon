# run-policy-operator

A Portal Operator that pre-loads multiple policies and exposes a single
RPC

```
run_policy(policy: "pick_up" | "put_down" | ..., side: "left" | "right")
```

to switch between them and take over the active-operator slot. Two
policy types are supported:

* **ACT policies** trained via
  `utils/data_collection/policies/act/train.py`. Each tick runs an ACT
  forward pass over the current camera + state.
* **Scripted policies** that play back a randomly-selected episode's
  recorded action sequence from a LeRobot v3 dataset. No model, no
  cameras read. Useful for motions like `put_down` where a learned
  policy is overkill and a prerecorded trajectory is enough.

The `side` parameter only affects scripted policies. `side="left"`
flips the sign of the first joint (shoulder_pan, the base rotation) on
every emitted action so a recording captured on the right mirrors to
the left. ACT policies ignore `side`.

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

Point the operator at one or more checkpoints (ACT) and/or scripted
datasets. Common setup: `pick_up` as ACT, `put_down` as scripted
playback.

```bash
# Env vars (handy for two well-known policies):
PICK_UP_CHECKPOINT=/path/to/pick_up/checkpoints/050000/pretrained_model \
PUT_DOWN_SCRIPT=/path/to/utils/data_collection/data/binhpham/spc-mock-put-down \
uv run python run_policy_operator.py

# Or repeatable --policy / --script flags (any name=path pair):
uv run python run_policy_operator.py \
    --policy pick_up=/path/to/pick_up/.../pretrained_model \
    --script put_down=/path/to/spc-mock-put-down
```

`--script` paths are LeRobot v3 dataset directories (the parent of
`data/chunk-*/file-*.parquet`). All episodes are loaded into memory at
startup; each `run_policy()` call resets to a random one.

The operator joins the configured LiveKit room as
`run-policy-operator` and parks until a peer calls
`run_policy(...)`. On call it:

1. preempts any in-flight `run_policy()` (the previous caller's RPC
   returns `{"status": "preempted", "by": "<new-name>"}`),
2. resets the chosen policy (ACT: clear chunk / temporal ensembler;
   scripted: pick a fresh random episode and arm a per-run duration
   watchdog at `episode_length / fps + 2 s`),
3. claims active-operator control (serialized against pending
   releases so back-to-back invocations can't leave the slot empty),
4. drives the robot with that policy until a terminator fires
   (scripted: end of episode or watchdog; ACT: sparse-reward streak
   and/or configured duration).

From another operator (e.g. a controller / orchestrator):

```python
# Bare-string form: defaults side="right".
result = await op.perform_rpc(
    "run_policy", "pick_up", destination="run-policy-operator",
)

# JSON form: pass a side to a scripted policy.
import json
result = await op.perform_rpc(
    "run_policy",
    json.dumps({"policy": "put_down", "side": "left"}),
    destination="run-policy-operator",
)
# result -> '{"policy": "put_down",
#             "side": "left",
#             "cameras": [],
#             "active_operator": "run-policy-operator-...",
#             "episode_index": 17,
#             "frames": 263,
#             "status": "done",
#             "reason": "script_end"}'
```

## Flags

```
--policy NAME=PATH                # repeat per ACT checkpoint
--script NAME=PATH                # repeat per scripted dataset dir
--default-policy NAME             # pre-select on startup (env: RUN_POLICY_DEFAULT)
--no-temporal-ensemble            # chunk-replan instead of per-tick ensemble
--temporal-ensemble-coeff 0.01    # default; lower = smoother, laggier
```

If a name is given to both `--policy` and `--script`, `--script` wins.
`--reward` and `--duration` targeting a scripted name are ignored (the
scripted terminator is the recorded episode length).

`--default-policy` lets the tick loop warm-start with a chosen policy
already selected (so it's producing actions immediately, just like
`molmo_operator`), without claiming active-operator. A peer (controller
or teleop `c`) still flips the active-operator slot; the warm policy
just means there's no inference delay when they do.

## Notes

* All registered policies are held in memory; only the selected one
  runs a tick. Loading two ACT checkpoints on a single GPU is cheap
  (~hundreds of MB each); a Molmo-class VLA would be a different
  story. A scripted dataset is ~100 MB on disk and a few MB once
  decoded into per-episode action arrays.
* If an ACT checkpoint declares cameras that the robot doesn't
  publish, the tick loop waits for them rather than feeding zeros.
  ACT also requires a fresh observation each tick (state + frames);
  if the robot stops publishing, ACT pauses but doesn't park.
* Scripted policies declare no cameras and never read observations.
  They keep advancing on the local fps clock even if `robot.py`'s
  camera stream has hung — useful for unsticking the arm with a
  recorded `put_down` when the obs side is degraded. `in_reply_to_ts_us`
  falls back to `0` when there's no obs yet.
* Scripted dataset load is vectorized through pyarrow: ~5 ms for the
  51-episode / 13 k-row `spc-mock-put-down` dataset, so registering
  several scripted policies at startup is essentially free.
* State / action wire shape is the shared 7-dim leslider schema (6 arm
  `.pos` + `slider.vel`). ACT checkpoints from data_collection are
  trained on the same shape, and scripted datasets recorded by the
  same pipeline match it directly, so no shimming is needed.
* `side="left"` flips action index 0 (shoulder_pan / base rotation)
  before sending. Other joints are untouched. ACT policies ignore the
  parameter (you'll get a warning log if you pass `side="left"` to
  one).
