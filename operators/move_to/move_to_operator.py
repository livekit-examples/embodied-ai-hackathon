"""Operator side: AprilTag-driven `move_to` controller.

Joins the leslider room as an Operator peer. Caches the latest
``overhead_camera`` frame and the latest synced state from the robot's
synchronizer, and exposes one RPC method, ``move_to(description)``,
that:

  1. Asks Moondream's cloud ``point`` endpoint to find the described
     object in the current overhead frame.
  2. Self-claims active-operator control of the robot.
  3. Runs a tick-rate control loop:
     - Detects an AprilTag (the slider's fiducial) in the overhead
       frame.
     - Compares the tag's pixel-Y to the target object's pixel-Y.
     - Sends one Portal action per tick: 6 arm ``.pos`` mirrored from
       the latest observed state, plus ``slider.vel`` set to a small
       signed constant.
  4. Stops when ``|tag_y - target_y|`` is within the threshold
     (success), the AprilTag is missing too long (tag-lost), or the
     timeout elapses.
  5. Sends ``slider.vel = 0`` and releases active operator before
     returning, regardless of outcome.

The RPC is synchronous: it blocks until the loop terminates and
returns a JSON summary.

Wire contract notes:

* The local ``portal.yaml`` mirrors the robot's contract.
* ``action_subscription`` is a YAML knob that costs us nothing here
  (we don't subscribe a callback for actions).
* Concurrent ``move_to`` calls are serialized via an asyncio.Lock;
  the second call waits for the first to finish.

Environment knobs (in addition to LIVEKIT_* and MOONDREAM_API_KEY):

* ``SLIDER_APRILTAG_FAMILY``  default ``tagStandard41h12``
* ``SLIDER_APRILTAG_ID``      optional integer; if set, only that tag
                              ID is accepted as the slider fiducial.
* ``SLIDER_MOVE_VELOCITY``    raw motor units; default ``800``
                              (within the leslider's MIN/MAX teleop
                              velocity).
* ``SLIDER_MOVE_THRESHOLD_PX`` pixel tolerance on Y; default ``20``.
* ``SLIDER_MOVE_TIMEOUT_S``   loop timeout in seconds; default ``15``.
* ``SLIDER_TAG_MISS_LIMIT``   consecutive ticks without a tag before
                              giving up; default ``30`` (~1s @ 30Hz).
* ``SLIDER_VEL_INVERT``       flip the sign of ``slider.vel`` once if
                              the rig drives the wrong way; default
                              ``false``.

Usage::

    cp .env.example .env  # set LIVEKIT_* and MOONDREAM_API_KEY
    uv sync               # picks up the new opencv + apriltag deps
    uv run move_to_operator.py

Then from any peer in the same room::

    result = await op.perform_rpc("move_to", "blue cube",
                                   destination="move-to-operator")
    # result -> '{"description": "blue cube",
    #             "target": {"x": 0.43, "y": 0.61,
    #                        "width": 640, "height": 480},
    #             "reached": true, "iterations": 47,
    #             "elapsed_s": 1.57, "final_tag_y_px": 291.2,
    #             "reason": "reached"}'
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
from typing import Optional

import cv2
import moondream as md
import numpy as np
from PIL import Image
from pupil_apriltags import Detector

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    RpcError,
    RpcInvocationData,
    frame_bytes_to_numpy_rgb,
)

from _common import (
    env_bool,
    env_int,
    env_str,
    load_env,
    mint_token,
    pace,
    required_env,
)

IDENTITY = "move-to-operator"
CAMERA = "overhead_camera"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

# 6 arm motor positions, in the wire order from portal.yaml. The 7th
# state field (`slider.vel`) is not mirrored — we drive it.
ARM_POS_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

logger = logging.getLogger(__name__)


def _detect_slider_tag(
    detector: Detector, rgb: np.ndarray, want_id: Optional[int]
) -> Optional[tuple[float, float]]:
    """Run the AprilTag detector on an RGB frame, return the chosen
    tag's pixel center ``(cx, cy)`` or ``None`` if no usable tag.

    When ``want_id`` is set, only tags with that ID are considered.
    Otherwise the lowest-numbered detected tag wins (deterministic
    when the camera sees more than one tag).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray)
    if want_id is not None:
        detections = [d for d in detections if d.tag_id == want_id]
    if not detections:
        return None
    d = min(detections, key=lambda x: x.tag_id)
    cx, cy = d.center
    return float(cx), float(cy)


async def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)
    api_key = required_env("MOONDREAM_API_KEY")
    fps = env_int("PORTAL_FPS", 30)

    # Control-loop knobs. Read once at startup; tweak via .env and
    # restart the operator to change them.
    move_velocity = env_int("SLIDER_MOVE_VELOCITY", 800)
    move_threshold_px = env_int("SLIDER_MOVE_THRESHOLD_PX", 20)
    move_timeout_s = env_int("SLIDER_MOVE_TIMEOUT_S", 15)
    tag_miss_limit = env_int("SLIDER_TAG_MISS_LIMIT", 30)
    vel_invert = env_bool("SLIDER_VEL_INVERT", default=False)
    tag_family = env_str("SLIDER_APRILTAG_FAMILY", "tagStandard41h12")
    raw_tag_id = os.environ.get("SLIDER_APRILTAG_ID", "").strip()
    want_tag_id: Optional[int] = int(raw_tag_id) if raw_tag_id else None

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    # Built once at startup so the per-RPC path is just an HTTP call,
    # not an SDK init. The client is thread-safe enough for our use:
    # we only ever call it from `asyncio.to_thread`, one RPC at a time.
    model = md.vl(api_key=api_key)

    # AprilTag detector reused across RPCs and loop iterations.
    # `quad_decimate=1.0` keeps full resolution; the slider tag is
    # small in the overhead frame and we'd rather pay CPU than miss
    # detections.
    tag_detector = Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
    )
    print(
        f"[move-to-op] apriltag detector ready: family={tag_family!r} "
        f"want_id={want_tag_id}"
    )

    # Updated on every synchronized obs. One-writer / one-reader: the
    # Portal callback (already hopped onto the asyncio loop) writes,
    # the RPC handler reads. No lock needed for atomic rebinds on
    # CPython.
    latest_overhead: Optional[np.ndarray] = None
    latest_state: dict[str, float] = {}
    latest_obs_ts_us: int = 0

    def on_observation(obs: Observation) -> None:
        nonlocal latest_overhead, latest_state, latest_obs_ts_us
        f = obs.frames.get(CAMERA)
        if f is not None:
            latest_overhead = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
        latest_state = dict(obs.state)
        latest_obs_ts_us = obs.timestamp_us

    op.on_observation(on_observation)

    # Serialize concurrent move_to calls. A second RPC waits behind
    # the first instead of fighting it for active-operator control.
    move_lock = asyncio.Lock()

    async def move_to(data: RpcInvocationData) -> str:
        # Accept either a bare string ("blue cube") or a JSON object
        # ({"description": "blue cube"}). The latter leaves room for
        # extra fields later (per-call camera, threshold, ...) without
        # changing the wire shape.
        payload = (data.payload or "").strip()
        description = payload
        if payload.startswith("{"):
            try:
                description = str(json.loads(payload).get("description", "")).strip()
            except json.JSONDecodeError:
                raise RpcError.Error(
                    code=1400,
                    message="payload was not valid JSON; pass a bare string instead",
                    data=None,
                )
        if not description:
            raise RpcError.Error(
                code=1400, message="empty description", data=None
            )

        frame = latest_overhead
        if frame is None:
            raise RpcError.Error(
                code=1404,
                message=f"no '{CAMERA}' frame yet; is robot.py running?",
                data=None,
            )
        # Refuse early if we have no state to mirror back. Without it
        # the action would read ".pos = 0" for every arm joint and
        # snap the arm to its zero pose, which is unsafe.
        if not all(k in latest_state for k in ARM_POS_KEYS):
            missing = [k for k in ARM_POS_KEYS if k not in latest_state]
            raise RpcError.Error(
                code=1409,
                message=f"no robot state yet (missing: {missing})",
                data=None,
            )

        h, w, _ = frame.shape
        image = Image.fromarray(frame)
        try:
            result = await asyncio.to_thread(model.point, image, description)
        except Exception as e:  # noqa: BLE001
            logger.exception("moondream point() failed")
            raise RpcError.Error(
                code=1502,
                message=f"moondream error: {type(e).__name__}: {e}",
                data=None,
            )

        points = result.get("points") or []
        if not points:
            raise RpcError.Error(
                code=1404,
                message=f"'{description}' not found in current frame",
                data=None,
            )

        p = points[0]
        target_xn = float(p["x"])
        target_yn = float(p["y"])
        target_y_px = target_yn * h
        print(
            f"[move-to-op] move_to('{description}') -> "
            f"target=({target_xn:.3f},{target_yn:.3f}) y_px={target_y_px:.1f}"
        )

        # One mover at a time. Queueing here is preferable to
        # rejecting a coincident call, since callers usually want
        # serial moves anyway.
        async with move_lock:
            me = op.local_identity()
            await op.set_active_operator(me)
            print(f"[move-to-op] active operator -> {me}")

            reached = False
            reason = "timeout"
            iterations = 0
            consecutive_misses = 0
            final_tag_y_px: Optional[float] = None
            t0 = time.monotonic()

            try:
                async for _ in pace(fps):
                    iterations += 1

                    if time.monotonic() - t0 > move_timeout_s:
                        reason = "timeout"
                        break

                    cur_frame = latest_overhead
                    if cur_frame is None:
                        consecutive_misses += 1
                        if consecutive_misses > tag_miss_limit:
                            reason = "tag_lost"
                            break
                        continue

                    tag = _detect_slider_tag(tag_detector, cur_frame, want_tag_id)
                    if tag is None:
                        consecutive_misses += 1
                        if consecutive_misses > tag_miss_limit:
                            reason = "tag_lost"
                            break
                        continue
                    consecutive_misses = 0
                    final_tag_y_px = tag[1]

                    delta_y_px = tag[1] - target_y_px
                    if abs(delta_y_px) < move_threshold_px:
                        reached = True
                        reason = "reached"
                        break

                    # Bang-bang: a single magnitude, sign chosen by
                    # which side of the target the tag is on. On this
                    # rig positive `slider.vel` drives the tag DOWN
                    # on the overhead image (increasing pixel Y), so
                    # when the tag is above the target (delta_y_px <
                    # 0) we send a positive velocity to push it down
                    # toward the target. SLIDER_VEL_INVERT lets a
                    # different rig flip the convention without code
                    # changes.
                    direction = -1.0 if delta_y_px > 0 else 1.0
                    if vel_invert:
                        direction = -direction
                    vel = direction * move_velocity

                    action = {
                        k: float(latest_state[k]) for k in ARM_POS_KEYS
                    }
                    action["slider.vel"] = float(vel)
                    op.send_action(
                        action,
                        timestamp_us=int(time.time() * 1_000_000),
                        in_reply_to_ts_us=latest_obs_ts_us,
                    )

                    if iterations % fps == 0:
                        print(
                            f"[move-to-op] tick {iterations:>4}: "
                            f"tag_y={tag[1]:.1f} target_y={target_y_px:.1f} "
                            f"delta={delta_y_px:+.1f} vel={vel:+.0f}"
                        )
            finally:
                # Always brake and release control. If we hit an
                # exception mid-loop the action stream would otherwise
                # carry whatever sign was last sent.
                if latest_state and all(k in latest_state for k in ARM_POS_KEYS):
                    stop_action = {
                        k: float(latest_state[k]) for k in ARM_POS_KEYS
                    }
                    stop_action["slider.vel"] = 0.0
                    op.send_action(
                        stop_action,
                        timestamp_us=int(time.time() * 1_000_000),
                        in_reply_to_ts_us=latest_obs_ts_us,
                    )
                try:
                    await op.set_active_operator(None)
                except Exception:
                    logger.exception("set_active_operator(None) failed")

            elapsed_s = time.monotonic() - t0
            print(
                f"[move-to-op] move_to done: reason={reason} reached={reached} "
                f"iterations={iterations} elapsed={elapsed_s:.2f}s"
            )

            return json.dumps({
                "description": description,
                "target": {
                    "x": target_xn,
                    "y": target_yn,
                    "width": w,
                    "height": h,
                },
                "reached": reached,
                "reason": reason,
                "iterations": iterations,
                "elapsed_s": elapsed_s,
                "final_tag_y_px": final_tag_y_px,
            })

    op.register_rpc_method("move_to", move_to)
    op.on_operator_joined(lambda i: print(f"[move-to-op] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[move-to-op] operator left: {i}"))

    print(f"[move-to-op] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()
    print(f"[move-to-op] connected as '{me}'; awaiting move_to() RPCs")

    try:
        # No tick loop. The control loop runs only inside an active
        # move_to RPC; otherwise we park.
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n[move-to-op] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
