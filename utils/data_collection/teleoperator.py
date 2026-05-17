"""Operator side: leslider leader + HITL recorder + rerun viewer.

Hardware lives in ``utils/leslider.py``. Rerun blueprint and per-tick
log helpers live in ``utils/rerun_view.py``. The hotkey listener lives
in ``utils/hotkeys.py``. The recorder lives in ``utils/recorder.py``.

Two YAML knobs (set in ``portal.yaml``) carry the wire-quality
contract for this side:

* ``action_subscription: true``. ``on_action`` fires for every action
  the robot's gate forwards, regardless of sender. Without it the
  recorder would only see actions we sent, not ones from a remote
  policy during HITL handoff. Each action carries ``Action.sender``
  so dataset rows can be labelled correctly.
* ``reuse_stale_frames: true``. The synchronizer reuses the last
  good frame on a track when the current state has no fresh frame
  within the sync window, instead of holding the state. State drives
  obs emission; frames are best-effort. Without this a transient
  camera drop punches holes in the recording. With it the dataset
  retains every tick, paying at most a one or two tick visual lag
  during loss.

Recording is driven by ``on_action``. One executed action gives one
dataset row, paired with the observation the operator was responding
to (anchored by ``in_reply_to_ts_us`` when the operator passed it,
otherwise by the action's send timestamp). With reuse_stale_frames on,
the sync delay is small and predictable, and actions always arrive at
this operator after the obs they reference. So a simple "find newest
obs <= target_ts" reconciliation is correct for steady state. The
``PORTAL_HITL_MAX_OBS_AGE_MS`` threshold (default 100ms) covers the
corner case where a state packet's wire transport is jittered enough
to flip the order; mismatched pairs are dropped and counted.

Hotkeys:
  c: cycle the active operator through ``[self, *remote_operators]``,
     wrapping back to self.
  r: toggle episode recording.
  [: discard the in-flight episode.
  x: clean quit.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from collections import deque
from typing import Optional

import numpy as np
import rerun as rr

from livekit.portal import (
    Action,
    Observation,
    Operator,
    OperatorConfig,
    frame_bytes_to_numpy_rgb,
)

from _common import env_int, env_str, load_env, mint_token, pace, required_env
from utils.hotkeys import Hotkeys
from utils.leslider import CAMERAS, build_leader
from utils.recorder import DatasetRecorder, build_from_frame
from utils.rerun_view import (
    build_blueprint,
    log_action,
    log_metrics,
    log_observation,
    log_recording,
)

IDENTITY = "teleoperator"
CONFIG_PATH = pathlib.Path(__file__).parent / "portal.yaml"

logger = logging.getLogger(__name__)


def _obs_to_record(obs: Observation) -> dict:
    """Reconstitute the recorder's flat observation dict from a synced
    ``Observation``: state scalars + decoded RGB ndarrays per camera."""
    out: dict = dict(obs.state)
    for cam in CAMERAS:
        f = obs.frames.get(cam)
        if f is not None:
            out[cam] = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
    return out


async def cycle_active_operator(op: Operator, me: str | None) -> None:
    """`c`: cycle active operator through ``[self, *others]`` and wrap
    back to self. With no remote operator connected the ring is just
    ``[self]``, so pressing `c` claims control for self — the common
    case during solo teleop before any policy has joined, when the
    active operator is still ``None`` and the robot is dropping our
    actions."""
    ring = [me, *op.operators()]
    current = op.active_operator()
    try:
        idx = ring.index(current)
    except ValueError:
        # Current active operator dropped from the ring (e.g. disconnected
        # mid-session) or was never set. Snap to self.
        idx = -1
    nxt = ring[(idx + 1) % len(ring)]
    await op.set_active_operator(nxt)
    print(f"[teleop] active operator → {nxt}")


async def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    fps = env_int("PORTAL_FPS", 30)
    max_obs_age_us = env_int("PORTAL_HITL_MAX_OBS_AGE_MS", 100) * 1000

    leader = build_leader()
    op = Operator(cfg)

    state_field_names = [f.name for f in cfg.state_schema]
    action_field_names = [f.name for f in cfg.action_schema]

    # Latest synced observation for the rerun mirror and recorder
    # construction. Both are written by Portal callbacks (already on the
    # asyncio loop) and read on the main tick. One-writer / one-reader.
    latest_obs: Optional[Observation] = None
    latest_state: dict = {}
    latest_frames: dict[str, np.ndarray] = {}
    latest_executed: Optional[Action] = None

    # Ring buffer of recent observations. Sized for ~500ms of history at
    # 30Hz: enough to align an action against the obs the operator was
    # responding to even with sync jitter, but bounded so memory doesn't
    # grow during a long session.
    obs_history: deque[Observation] = deque(maxlen=16)
    skipped_frames = 0

    recorder: Optional[DatasetRecorder] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs, latest_state
        latest_obs = obs
        latest_state = dict(obs.state)
        for cam in CAMERAS:
            f = obs.frames.get(cam)
            if f is not None:
                latest_frames[cam] = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
        obs_history.append(obs)

    def on_action(action: Action) -> None:
        # The action stream drives the recorder. One action = one row,
        # paired with the obs the operator was responding to.
        nonlocal latest_executed, skipped_frames
        latest_executed = action

        if recorder is None or not recorder.is_recording:
            return

        # Policies pass `in_reply_to_ts_us` to anchor the action to a
        # specific observation. The teleop leader doesn't, so fall back
        # to the action's own send timestamp as a "what the human saw
        # when they moved" approximation.
        target_ts = action.in_reply_to_ts_us or action.timestamp_us

        aligned: Observation | None = None
        for o in reversed(obs_history):
            if o.timestamp_us <= target_ts:
                aligned = o
                break
        if aligned is None:
            return  # episode start, no obs old enough yet

        if target_ts - aligned.timestamp_us > max_obs_age_us:
            skipped_frames += 1
            return  # observation stream stalled; would mislabel the row

        recorder.push_frame(
            _obs_to_record(aligned),
            {k: float(v) for k, v in action.values.items()},
        )
        for err in recorder.poll_errors():
            print(f"[teleop] recorder error: {err}")

    op.on_observation(on_observation)
    op.on_action(on_action)
    op.on_active_operator_changed(
        lambda i: print(f"[teleop] active operator now: {i}")
    )
    op.on_operator_joined(lambda i: print(f"[teleop] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[teleop] operator left: {i}"))

    leader.connect()
    print(f"[teleop] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()

    print(f"[teleop] connected as '{me}' @ {fps} fps")
    print("[teleop] hotkeys: c=cycle operator  r=record  [=discard  x=quit")
    print("[teleop] slider keys: ←/→ drive  ↑/↓ trim  space stop  (handled by leader)")

    rr.init(f"portal-hitl-{room}", spawn=True)
    rr.send_blueprint(build_blueprint())

    hotkeys = Hotkeys()
    hotkeys.start()
    quit_requested = False

    try:
        async for tick in pace(fps):
            # Hotkeys.
            for key in hotkeys.pop_pressed():
                if key == "c":
                    await cycle_active_operator(op, me)
                elif key == "r":
                    if recorder is None:
                        print("[teleop] waiting for first observation before recording")
                    elif recorder.is_recording:
                        recorder.end_episode()
                        print(f"[teleop] episode {recorder.episode_count - 1} saving in background")
                    else:
                        # Reset the obs ring before starting so the first
                        # row of the new episode pairs against an obs from
                        # within the episode, not from before.
                        obs_history.clear()
                        skipped_frames = 0
                        recorder.start_episode()
                        print(f"[teleop] episode {recorder.episode_count} recording")
                elif key == "[":
                    if recorder is not None:
                        if recorder.discard_episode():
                            print("[teleop] episode discarded")
                        else:
                            print("[teleop] nothing to discard (not recording; saved episodes are kept)")
                elif key == "x":
                    quit_requested = True
            if quit_requested:
                break

            # Drive the wire. Always send, even when not active; the
            # robot's gate drops mismatched-sender actions, and keeping
            # the leader in sync means takeover from a policy is instant.
            op.send_action(leader.get_action(), timestamp_us=int(time.time() * 1_000_000))

            # Lazy-build the recorder once we know the wire camera
            # resolution. The YAML doesn't pin width/height; the
            # follower's camera config does.
            if recorder is None and len(latest_frames) >= len(CAMERAS):
                recorder = build_from_frame(
                    repo_id=env_str("PORTAL_HITL_DATASET_REPO_ID", "local/portal-hitl"),
                    fps=fps,
                    state_field_names=state_field_names,
                    action_field_names=action_field_names,
                    cameras=CAMERAS,
                    frame=latest_frames[CAMERAS[0]],
                    task=env_str("PORTAL_HITL_DATASET_TASK", "leslider manipulation"),
                    root=os.environ.get("PORTAL_HITL_DATASET_ROOT") or None,
                )
                print(f"[teleop] recorder ready (episodes so far={recorder.episode_count})")

            # Mirror to rerun. Reads the latest snapshots; rerun is a
            # visualization, not the dataset, so duplicate frames at
            # tick rate don't matter.
            log_observation(latest_obs, latest_state, latest_frames)
            log_action(latest_executed, me)
            log_recording(recorder)
            if tick % fps == 0 and (m := op.metrics()) is not None:
                log_metrics(m)

            # Periodic skip surface: every 5s, report and reset.
            if tick % (fps * 5) == 0 and skipped_frames > 0:
                print(f"[teleop] skipped {skipped_frames} frames in last 5s (obs older than {max_obs_age_us // 1000}ms)")
                skipped_frames = 0

    except KeyboardInterrupt:
        print("\n[teleop] stopping ...")
    finally:
        hotkeys.stop()
        if recorder is not None:
            try:
                recorder.finalize()
            except Exception:
                logger.exception("recorder.finalize failed")
        try:
            await op.disconnect()
        finally:
            op.close()
            leader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
