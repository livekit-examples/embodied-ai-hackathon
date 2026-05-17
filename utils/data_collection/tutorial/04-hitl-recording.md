# 4. HITL recording

Recording is the slightly subtle part. The naive version,
`leader.get_action()` paired with `op.get_observation()` on every
tick, fails in a few ways once you look closely:

1. Tick rate beats observation arrival. Same `(obs, action)` row
   written N times, bit-identical consecutive frames.
2. Observations arrive faster than ticks. Some frames silently lost.
3. Misaligned pairs. The recorded action was generated in response to
   an observation different from the one stored next to it.
4. Stale slot at episode start. First rows pair against a previous
   episode's data.

Fixing all of these comes down to two YAML flags, two Portal ideas,
and one design choice in the recording loop.

## YAML flags that matter for data quality

```yaml
# portal.yaml
reuse_stale_frames: true
action_subscription: true
```

`reuse_stale_frames`: when a state sample arrives at the operator's
synchronizer with no fresh frame inside the sync window, reuse the
last good frame instead of dropping the state. State drives obs
emission; frames are best-effort. Without this, a transient camera
drop punches holes in the recording (state without obs); with it the
dataset retains every tick, paying at most a one or two tick visual
lag during loss. For training data this is the right tradeoff: a
slightly stale frame is more useful than a missing row.

`action_subscription`: tells the operator role to receive the executed
action stream over the wire. With it on, `op.on_action(...)` fires
every time the robot's gate forwards an action to the arm:

- Actions you sent when you were the active operator.
- Actions a remote policy sent when it was the active operator.
- Nothing while no operator is active (the gate dropped them).

The data in this callback is exactly what the robot applied.

## Action.sender

Each `Action` carries a `sender` field set at gate time on the robot:

```python
def on_action(action):
    print(action.sender, action.values)
```

Use this to label rows. The active-operator pointer can race with
handoff between two `set_active_operator` calls, but the robot stamps
`sender` at the gate so it can't be wrong by the time you see it.
Prefer `action.sender` over `op.active_operator()` for labelling.

## Drive recording from on_action, not from the tick

For training data we want one row per executed action, paired with
the observation the operator was responding to. The action stream
drives the writer:

```python
obs_history: deque[Observation] = deque(maxlen=16)

def on_observation(obs):
    obs_history.append(obs)
    # ...

def on_action(action):
    if not recorder.is_recording:
        return

    # Find the obs the operator was responding to. Policies pass
    # `in_reply_to_ts_us`; the leader doesn't, so fall back to the
    # action's own send timestamp as an approximation.
    target_ts = action.in_reply_to_ts_us or action.timestamp_us

    aligned = None
    for o in reversed(obs_history):
        if o.timestamp_us <= target_ts:
            aligned = o
            break
    if aligned is None:
        return  # episode start, no obs old enough yet
    if target_ts - aligned.timestamp_us > MAX_OBS_AGE_US:
        return  # synchronization stalled, would mislabel

    recorder.push_frame(_obs_to_record(aligned), dict(action.values))
```

What this gives:

- One row per executed action. No duplicates.
- No silently lost observations: every `on_observation` lands in the
  ring; obs that aren't matched at arrival time are picked up by a
  later `on_action`.
- `(obs, action)` pairs the agent's actual decision with the obs it
  was looking at, not whatever happened to be in `latest_obs` at
  write time.
- Stale-stream rejection. If the obs stream stalls relative to the
  action stream, the threshold check catches it instead of silently
  recording a wrong pair.
- Episode boundaries clean. `obs_history.clear()` on `start_episode`
  ensures the first row of a new episode pairs against an obs from
  inside the episode, not from before.

## Why a simple "find newest obs ≤ target_ts" works here

It looks fragile but it isn't, given the YAML flags above. Walk the
timing:

```
obs(T) at teleop      ≈ T + RTT_tr/2 + small_smoothing
action(T) at teleop   = T + RTT_pr + D_policy_sync + T_inf + RTT_tr/2
```

With `reuse_stale_frames: true`, the synchronizer doesn't wait for a
matching frame, so `small_smoothing` at both operators is tiny and
roughly equal (`D_policy_sync` ≈ `small_smoothing` since both run
the same YAML). The difference between action arrival and obs
arrival reduces to `RTT_pr + T_inf`, always positive: the action
always lands after the obs it references.

The only residual case is state-packet jitter on the teleop's wire
specifically (the data channel is reliable but not infinitely so).
That's what the `MAX_OBS_AGE` threshold is for: catch the rare
mis-ordered pair and drop it rather than write a wrong row.

Without `reuse_stale_frames`, sync would block on missing frames and
those bounds break down; a dual-buffer pending-action queue would be
needed. With it on, the simple lookup is the right tool.

## Why not drive from on_observation

You could. But the action stream is the throughput-limiting one:
human-driven leader emits one action per tick, and slow policies emit
fewer. Driving from observations gives 30 rows/sec even when the
action stream is at 10Hz, which means most rows would carry the same
action repeated. That isn't gaps in time, but it bloats the dataset
with redundancy and obscures the agent's actual decision rate.
Driving from actions keeps row count honest with what the agent
actually produced.

The flip side is that driving from on_action means one row per
action, even if the policy is slow, so a 10Hz policy produces ~10
rows/sec into a dataset whose `fps` metadata says 30. lerobot reads
in arrival order; missing rows appear as dataset gaps, not
mistimestamped data, which is the right failure mode.

## Recording while a policy drives

The teleoperator must be running for any recording to happen, even
when a policy is the active operator. Workflow:

```
1. uv run robot.py
2. uv run teleoperator.py        # press 'r' to start recording
3. uv run policies/act/inference.py
4. On teleop window: 'c' twice to hand control to the policy.
   Recording continues; rows now show Action.sender == "policy-act".
```

The policy doesn't record itself. It only sends actions; the robot's
gate forwards them to the teleop's `on_action` callback (because of
`action_subscription: true`), and the teleop's recorder pairs each
one with the synced obs the policy was responding to (via
`in_reply_to_ts_us`).

If you want headless policy-only recording (no human present at all),
peel a "passive recorder" operator out of `teleoperator.py` that
doesn't drive a leader and never claims control. It is not in this
repo, but the implementation is just the recorder slice without the
hardware.

## What the recorder does

`utils/recorder.py` wraps `LeRobotDataset` with three things:

- Off-thread `save_episode`. The hot path never blocks on parquet or
  video encoding.
- Resume. If `data/<repo_id>/meta/tasks.parquet` exists, reuse the
  dataset and append. Otherwise create.
- `start_episode` / `end_episode` / `discard_episode` toggled by `r`
  and `[`.

It is intentionally not Portal-aware. Hand it `observation` and
`action` as flat dicts; it shapes them into a `LeRobotDataset` frame.

Next: [05. Handoff](05-handoff.md).
