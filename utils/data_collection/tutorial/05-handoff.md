# 5. Handoff

Multiple operators can be in the same room. The robot accepts actions
from exactly one of them at a time: the **active operator**. Anyone
else who calls `op.send_action(...)` has their packets dropped at the
gate.

## The active operator pointer

The robot owns it. The pointer is broadcast as a participant attribute
(`lk.portal.active_operator`) so every peer can read it without a round
trip:

```python
op.active_operator()                 # current pointer (Optional[str])
await op.set_active_operator("foo")  # set; on operator side this is an RPC
await op.set_active_operator(None)   # clear; arm idles
```

When the pointer changes, every peer's `on_active_operator_changed`
callback fires:

```python
op.on_active_operator_changed(lambda i: print("active now:", i))
```

The callback receives the new identity, or `None` if the pointer was
cleared.

## Discovering peers

Operators can list each other:

```python
op.local_identity()   # your own identity (after connect())
op.operators()        # other operators currently in the room
op.robot_identity()   # the robot's identity
```

`operators()` excludes self by definition, so the list is "everyone
else who could be driving."

## The toggle hotkey

`teleoperator.py` binds the `c` key to a two-state toggle:

```python
if op.active_operator() == me:
    others = op.operators()
    nxt = others[0] if others else me   # hand off to policy
else:
    nxt = me                            # claim self
await op.set_active_operator(nxt)
```

Either the human is driving or the policy is. The default state on
connect is `None` (neither), so the first `c` press claims the human.
After that, every press flips control between the two roles. If no
policy is connected, toggling away from self is a no-op since there
is no other operator to hand to.

## What happens at the gate

When the robot's gate sees an inbound action:

1. Read sender identity from the packet metadata.
2. Compare to the local active-operator pointer.
3. If they match, deliver to `on_action` (and to subscribed operators,
   see [04. HITL recording](04-hitl-recording.md)).
4. If they don't, drop. No callback fires.

The pointer is sticky across disconnect. If the active operator drops
out of the room, the robot keeps the pointer pinned so a same-identity
reconnect resumes control without re-claiming. Clear it explicitly
(set to `None`) if you want the arm to idle on disconnect.

## Latency closure

Operators that respond to observations should pass the obs timestamp
back when they send:

```python
op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)
```

The robot diffs that against its current wall clock and surfaces the
result as `metrics.policy.e2e_us_p50` / `_p95`. That is the true
observation-to-action delay end to end, including transport and sync
buffering. Network RTT alone tells you nothing about whether the
policy is actually keeping up.

The teleoperator does not pass `in_reply_to_ts_us` because the leader
arm produces actions independently of the observation stream. The
policies do, because their actions are responses to observations.

Next: [06. Plugging in a policy](06-plugging-in-policies.md).
