"""Operator side: AprilTag-driven `move_to` controller.

Joins the leslider room as an Operator peer. Caches the latest
``overhead_camera`` frame and the latest synced state from the robot's
synchronizer, and exposes three RPC methods.

``move_to(description)``:

  1. Asks Moondream's cloud ``point`` endpoint to find the described
     object in the current overhead frame.
  2. Self-claims active-operator control of the robot.
  3. Runs a tick-rate control loop:
     - Detects an AprilTag (the slider's fiducial) in the overhead
       frame.
     - Compares the tag's pixel-Y to the target object's pixel-Y.
       (The overhead camera is mounted so the slider's travel axis
       runs up/down in the image; we drive Y, not X.)
     - Sends one Portal action per tick: 6 arm ``.pos`` mirrored from
       the latest observed state, plus ``slider.vel`` set to a small
       signed constant.
  4. Stops when ``|tag_y - target_y|`` is within the threshold
     (success), the AprilTag is missing too long (tag-lost), or the
     timeout elapses.
  5. Sends ``slider.vel = 0`` and releases active operator before
     returning, regardless of outcome.

``reset_pose()``: drives the 6 arm joints back to a configured rest
pose with per-tick ramping, holds ``slider.vel = 0`` throughout, and
returns when every joint sits within tolerance (or on timeout). The
target pose is fully env-configurable per joint (see ``RESET_POSE_*``
below) so each rig can dial in its own safe rest position; the
default values in ``RESET_POSE_DEFAULTS`` are sampled from the
leslider's known parked pose and are a good starting point for that
rig.

``release_gripper()``: ramps ``gripper.pos`` open to a fixed released
pose (``90``) while mirroring every other arm joint from the latest
observation and holding ``slider.vel = 0``. Reuses the ``reset_pose``
ramp knobs (``RESET_POSE_MAX_STEP`` / ``RESET_POSE_TOLERANCE`` /
``RESET_POSE_TIMEOUT_S``) so per-rig tuning lives in one place.

All RPCs are synchronous: they block until the loop terminates and
return a JSON summary. They share an asyncio.Lock so a reset or
release that arrives mid-move queues behind the move (and vice versa)
rather than fighting it for active-operator control.

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
* ``SLIDER_REACHED_HOLD_TICKS`` consecutive ticks with
                              ``|delta| < threshold`` required before
                              the loop declares success. A single
                              spurious AprilTag detection that lands
                              near the target should not be enough to
                              stop the move, so we require the slider
                              to actually sit inside the window for
                              this many ticks. While we're waiting,
                              the loop sends ``slider.vel = 0`` so
                              the slider stops rather than coasting
                              past. Default ``3``.
* ``SLIDER_VEL_INVERT``       flip the sign of ``slider.vel`` once if
                              the rig drives the wrong way; default
                              ``false``.
* ``SLIDER_TAG_OFFSET_Y_PX``  signed pixel offset from the AprilTag
                              center to the gripper center along the
                              image Y axis; added to the tag's Y so
                              the gripper (not the tag) lands on the
                              target. Default ``0``. Used as a flat
                              offset when the perspective-scaled
                              anchors below are not set.
* ``SLIDER_TAG_OFFSET_NEAR_Y_PX`` / ``SLIDER_TAG_OFFSET_NEAR_PX`` and
  ``SLIDER_TAG_OFFSET_FAR_Y_PX`` / ``SLIDER_TAG_OFFSET_FAR_PX``
                              optional two-point calibration of how
                              the tag→gripper pixel offset varies with
                              the tag's image Y. A fixed physical
                              offset projects to fewer pixels the
                              further the slider sits from the
                              camera, so the offset has to scale with
                              Y. Provide ``(image_y, offset_px)`` at
                              two reference positions (typically one
                              near the camera, one far) and the
                              operator linearly interpolates between
                              them, clamping outside the range. If
                              any of the four is unset, the operator
                              falls back to the flat
                              ``SLIDER_TAG_OFFSET_Y_PX``.

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

    result = await op.perform_rpc("reset_pose", "",
                                   destination="move-to-operator")
    # result -> '{"target": {"shoulder_pan.pos": 0.0, ...},
    #             "final":  {"shoulder_pan.pos": 0.1, ...},
    #             "reached": true, "iterations": 22,
    #             "elapsed_s": 0.74, "reason": "reached"}'
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

# Per-joint env override names for the `reset_pose` target. Kept aligned
# with ARM_POS_KEYS so iteration order matches the wire order.
RESET_POSE_ENV_KEYS: dict[str, str] = {
    "shoulder_pan.pos":  "RESET_POSE_SHOULDER_PAN",
    "shoulder_lift.pos": "RESET_POSE_SHOULDER_LIFT",
    "elbow_flex.pos":    "RESET_POSE_ELBOW_FLEX",
    "wrist_flex.pos":    "RESET_POSE_WRIST_FLEX",
    "wrist_roll.pos":    "RESET_POSE_WRIST_ROLL",
    "gripper.pos":       "RESET_POSE_GRIPPER",
}

# Per-joint defaults for the `reset_pose` target. Sampled from a
# parked-but-safe rest pose on the leslider rig. RESET_POSE_* env
# vars override these; setting one to an empty string keeps the
# default. Units are degrees on the SO-101 follower.
RESET_POSE_DEFAULTS: dict[str, float] = {
    "shoulder_pan.pos":   1.76,
    "shoulder_lift.pos": -99.87,
    "elbow_flex.pos":    90.73,
    "wrist_flex.pos":    73.01,
    "wrist_roll.pos":     0.04,
    "gripper.pos":       10,
}

logger = logging.getLogger(__name__)


def _build_reset_pose_target() -> dict[str, float]:
    """Resolve env into a per-joint target dict for ``reset_pose``.

    Any joint without a ``RESET_POSE_*`` override falls back to the
    rig-tuned default in ``RESET_POSE_DEFAULTS`` (sampled from a known
    safe rest pose on the leslider). Override per joint in ``.env`` if
    your rig parks somewhere different.
    """
    pose: dict[str, float] = {}
    for key, env_name in RESET_POSE_ENV_KEYS.items():
        raw = os.environ.get(env_name, "").strip()
        pose[key] = float(raw) if raw else RESET_POSE_DEFAULTS[key]
    return pose


def _build_tag_offset_fn() -> tuple[callable, str]:
    """Resolve env into a function ``offset_px(tag_y_px) -> float``.

    With perspective, a fixed physical tag→gripper offset projects to
    fewer pixels the further the slider sits from the camera. We model
    that as a linear function of the tag's image Y, calibrated from
    two anchors: ``(NEAR_Y_PX, NEAR_PX)`` and ``(FAR_Y_PX, FAR_PX)``.
    Outside the calibrated range we clamp rather than extrapolate, so
    a narrow calibration can't produce silly offsets at the edges of
    the workspace.

    Falls back to a flat ``SLIDER_TAG_OFFSET_Y_PX`` (default ``0``)
    when any of the four anchor values is missing.
    """
    raw = {
        k: os.environ.get(k, "").strip()
        for k in (
            "SLIDER_TAG_OFFSET_NEAR_Y_PX",
            "SLIDER_TAG_OFFSET_NEAR_PX",
            "SLIDER_TAG_OFFSET_FAR_Y_PX",
            "SLIDER_TAG_OFFSET_FAR_PX",
        )
    }
    if all(raw.values()):
        a_y = float(raw["SLIDER_TAG_OFFSET_NEAR_Y_PX"])
        a_o = float(raw["SLIDER_TAG_OFFSET_NEAR_PX"])
        b_y = float(raw["SLIDER_TAG_OFFSET_FAR_Y_PX"])
        b_o = float(raw["SLIDER_TAG_OFFSET_FAR_PX"])
        if a_y == b_y:
            raise RuntimeError(
                "SLIDER_TAG_OFFSET_NEAR_Y_PX and SLIDER_TAG_OFFSET_FAR_Y_PX "
                "must differ"
            )
        # Sort so lo_y < hi_y for clean clamping.
        if a_y <= b_y:
            lo_y, lo_o, hi_y, hi_o = a_y, a_o, b_y, b_o
        else:
            lo_y, lo_o, hi_y, hi_o = b_y, b_o, a_y, a_o
        span = hi_y - lo_y

        def fn(y: float) -> float:
            yc = lo_y if y < lo_y else (hi_y if y > hi_y else y)
            t = (yc - lo_y) / span
            return lo_o + t * (hi_o - lo_o)

        desc = (
            f"linear(y={lo_y:.0f}->off={lo_o:+.1f}, "
            f"y={hi_y:.0f}->off={hi_o:+.1f}, clamped)"
        )
        return fn, desc

    const = float(env_int("SLIDER_TAG_OFFSET_Y_PX", 0))
    return (lambda _y: const), f"constant({const:+.1f})"


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
    move_velocity = env_int("SLIDER_MOVE_VELOCITY", 6000)
    move_threshold_px = env_int("SLIDER_MOVE_THRESHOLD_PX", 20)
    move_timeout_s = env_int("SLIDER_MOVE_TIMEOUT_S", 20)
    tag_miss_limit = env_int("SLIDER_TAG_MISS_LIMIT", 30)
    reached_hold_ticks = env_int("SLIDER_REACHED_HOLD_TICKS", 3)
    vel_invert = env_bool("SLIDER_VEL_INVERT", default=False)
    tag_offset_fn, tag_offset_desc = _build_tag_offset_fn()
    tag_family = env_str("SLIDER_APRILTAG_FAMILY", "tag16h5")
    # Pixel-Y safety window for the slider tag. Outside this range
    # the loop refuses to push further (see at-limit check below).
    # Values were lifted from the prior X-axis tuning — retune for
    # the new camera orientation if the slider hits the rail before
    # the loop catches it.
    tag_min = 90
    tag_max = 287
    raw_tag_id = os.environ.get("SLIDER_APRILTAG_ID", "").strip()
    want_tag_id: Optional[int] = int(raw_tag_id) if raw_tag_id else None

    # `reset_pose` knobs. See module docstring for the per-joint
    # RESET_POSE_* overrides; the rest are loop-shape parameters.
    reset_pose_target = _build_reset_pose_target()
    reset_max_step = float(env_str("RESET_POSE_MAX_STEP", "2.0"))
    reset_tolerance = float(env_str("RESET_POSE_TOLERANCE", "5.0"))
    reset_timeout_s = env_int("RESET_POSE_TIMEOUT_S", 10)

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
    print(f"[move-to-op] tag offset model: {tag_offset_desc}")
    print(
        "[move-to-op] reset_pose target: "
        + ", ".join(f"{k}={reset_pose_target[k]:+.2f}" for k in ARM_POS_KEYS)
        + f" (max_step={reset_max_step}, tol={reset_tolerance}, "
        f"timeout={reset_timeout_s}s)"
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
            consecutive_inside = 0
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

                    # The AprilTag is mounted on the slider but not
                    # exactly above the gripper, so we shift the
                    # tag's Y by a pixel offset to get the gripper's
                    # effective image-space Y. Drive that, not the
                    # raw tag center, to the target. The offset
                    # scales with the tag's image Y because a fixed
                    # physical separation projects to fewer pixels
                    # the further the slider sits from the camera.
                    tag_offset_px = tag_offset_fn(tag[1])
                    gripper_y_px = tag[1] + tag_offset_px
                    delta_y_px = gripper_y_px - target_y_px

                    # Debounced "reached" check: require the slider to
                    # actually sit inside the threshold window for
                    # ``reached_hold_ticks`` consecutive frames before
                    # declaring success. A single in-window detection
                    # is not trustworthy here because (a) bang-bang
                    # control means the slider can streak through the
                    # window between samples, and (b) the AprilTag
                    # family in use (``tag16h5`` by default) has a
                    # high false-positive rate, so one spurious
                    # decode near the target Y can otherwise short-
                    # circuit the loop with the slider still far
                    # away. While we're inside the window but not
                    # yet confirmed, brake (``slider.vel = 0``) so
                    # the slider settles instead of coasting.
                    if abs(delta_y_px) < move_threshold_px:
                        consecutive_inside += 1
                        if consecutive_inside >= reached_hold_ticks:
                            reached = True
                            reason = "reached"
                            print(
                                f"[move-to-op] reached at iter "
                                f"{iterations}: tag_y={tag[1]:.1f} "
                                f"off={tag_offset_px:+.1f} "
                                f"grip_y={gripper_y_px:.1f} "
                                f"target_y={target_y_px:.1f} "
                                f"delta={delta_y_px:+.1f} "
                                f"hold={consecutive_inside}"
                            )
                            break
                        brake_action = {
                            k: float(latest_state[k]) for k in ARM_POS_KEYS
                        }
                        brake_action["slider.vel"] = 0.0
                        op.send_action(
                            brake_action,
                            timestamp_us=int(time.time() * 1_000_000),
                            in_reply_to_ts_us=latest_obs_ts_us,
                        )
                        continue
                    consecutive_inside = 0

                    # Refuse to drive the slider past its safe pixel-Y
                    # window. delta_y_px > 0 commands motion toward
                    # DECREASING tag Y (see direction logic below), so
                    # if the tag is already at/past `tag_min` we must
                    # not push further; symmetric on the high side at
                    # `tag_max`. The reached check above takes priority
                    # so a target inside the safe window can still be
                    # met right at the boundary.
                    if (tag[1] <= tag_min and delta_y_px > 0) or (
                        tag[1] >= tag_max and delta_y_px < 0
                    ):
                        print(
                            f"[move-to-op] tag at limit: tag_y={tag[1]:.1f} "
                            f"range=[{tag_min},{tag_max}] delta={delta_y_px:+.1f}"
                        )
                        reason = "at_limit"
                        break

                    # Bang-bang: a single magnitude, sign chosen by
                    # which side of the target the tag is on. The
                    # overhead camera is reoriented so the slider's
                    # travel axis runs up/down in the image. On this
                    # rig, positive `slider.vel` drives the tag toward
                    # DECREASING pixel Y, so when the tag sits below
                    # the target in the image (delta_y_px > 0) we send
                    # a positive velocity to push it up toward the
                    # target. SLIDER_VEL_INVERT lets a different rig
                    # (or flipped camera orientation) flip the
                    # convention without code changes.
                    direction = 1.0 if delta_y_px > 0 else -1.0
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
                            f"tag_y={tag[1]:.1f} off={tag_offset_px:+.1f} "
                            f"grip_y={gripper_y_px:.1f} "
                            f"target_y={target_y_px:.1f} "
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

    async def reset_pose(data: RpcInvocationData) -> str:
        # Ignore the payload. Keeping the signature accepting (and not
        # validating) data leaves room to grow into per-call overrides
        # without breaking existing callers.
        del data

        if not all(k in latest_state for k in ARM_POS_KEYS):
            missing = [k for k in ARM_POS_KEYS if k not in latest_state]
            raise RpcError.Error(
                code=1409,
                message=f"no robot state yet (missing: {missing})",
                data=None,
            )

        target = dict(reset_pose_target)
        start_state = {k: float(latest_state[k]) for k in ARM_POS_KEYS}
        print(
            "[move-to-op] reset_pose start: "
            + ", ".join(
                f"{k}: {start_state[k]:+.2f}->{target[k]:+.2f}"
                for k in ARM_POS_KEYS
            )
        )

        async with move_lock:
            me = op.local_identity()
            await op.set_active_operator(me)
            print(f"[move-to-op] active operator -> {me}")

            reached = False
            reason = "timeout"
            iterations = 0
            t0 = time.monotonic()
            final_state: dict[str, float] = dict(start_state)
            # Open-loop ramp seed: where we'll *say* the arm should be
            # next tick. Starting from the observed pose so the first
            # commanded delta is bounded by `reset_max_step`.
            commanded: dict[str, float] = dict(start_state)

            try:
                async for _ in pace(fps):
                    iterations += 1

                    if time.monotonic() - t0 > reset_timeout_s:
                        reason = "timeout"
                        break

                    # Convergence is gated on observed state so "reached"
                    # actually means the motor got there, not just that
                    # we *commanded* it there.
                    cur = {k: float(latest_state[k]) for k in ARM_POS_KEYS}
                    final_state = cur

                    # Per-joint ramp from the *last commanded* value, not
                    # `cur`. Ramping off `cur` makes the operator chase
                    # the arm: every tick the command sits only
                    # `reset_max_step` ahead of the observed position, so
                    # the motor only ever sees a ~reset_max_step error to
                    # close and its effective velocity collapses to its
                    # own response rate instead of `fps * reset_max_step`.
                    # Ramping the command open-loop lets the operator
                    # build a position lead the motor can actually chase
                    # at full speed; convergence is still bounded by
                    # `max_remaining` below.
                    next_action: dict[str, float] = {}
                    max_remaining = 0.0
                    for k in ARM_POS_KEYS:
                        diff = target[k] - commanded[k]
                        if abs(diff) <= reset_max_step:
                            commanded[k] = target[k]
                        elif diff > 0:
                            commanded[k] = commanded[k] + reset_max_step
                        else:
                            commanded[k] = commanded[k] - reset_max_step
                        next_action[k] = commanded[k]
                        max_remaining = max(max_remaining, abs(target[k] - cur[k]))
                    next_action["slider.vel"] = 0.0

                    op.send_action(
                        next_action,
                        timestamp_us=int(time.time() * 1_000_000),
                        in_reply_to_ts_us=latest_obs_ts_us,
                    )

                    if max_remaining <= reset_tolerance:
                        reached = True
                        reason = "reached"
                        print(
                            f"[move-to-op] reset_pose reached at iter "
                            f"{iterations}: max_remaining={max_remaining:.2f}"
                        )
                        break

                    if iterations % fps == 0:
                        print(
                            f"[move-to-op] reset tick {iterations:>4}: "
                            f"max_remaining={max_remaining:.2f}"
                        )
            finally:
                # Same brake-and-release contract as move_to: never leave
                # a stray slider velocity in the action stream and never
                # squat on active-operator control after the RPC ends.
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
                f"[move-to-op] reset_pose done: reason={reason} "
                f"reached={reached} iterations={iterations} "
                f"elapsed={elapsed_s:.2f}s"
            )

            return json.dumps({
                "target": target,
                "final": final_state,
                "reached": reached,
                "reason": reason,
                "iterations": iterations,
                "elapsed_s": elapsed_s,
            })

    async def release_gripper(data: RpcInvocationData) -> str:
        # Mirrors `latest_state` for every arm joint except the gripper,
        # which ramps open to a fixed "released" pose (90 deg on the
        # SO-101 follower). Same brake-and-release contract as the other
        # RPCs: serialized behind `move_lock`, takes active-operator
        # control, holds `slider.vel = 0`, and releases on exit. Reuses
        # the `reset_pose` ramp knobs (`reset_max_step`,
        # `reset_tolerance`, `reset_timeout_s`) so per-rig tuning lives
        # in one place.
        del data

        release_target = 50.0

        if not all(k in latest_state for k in ARM_POS_KEYS):
            missing = [k for k in ARM_POS_KEYS if k not in latest_state]
            raise RpcError.Error(
                code=1409,
                message=f"no robot state yet (missing: {missing})",
                data=None,
            )

        start_state = {k: float(latest_state[k]) for k in ARM_POS_KEYS}
        print(
            f"[move-to-op] release_gripper start: "
            f"gripper.pos {start_state['gripper.pos']:+.2f}->{release_target:+.2f}"
        )

        async with move_lock:
            me = op.local_identity()
            await op.set_active_operator(me)
            print(f"[move-to-op] active operator -> {me}")

            reached = False
            reason = "timeout"
            iterations = 0
            t0 = time.monotonic()
            final_state: dict[str, float] = dict(start_state)
            # Open-loop ramp seed for the gripper. Same rationale as in
            # `reset_pose`: ramping the command (not the observed pose)
            # builds a position lead the motor can chase at full speed.
            commanded_gripper = float(start_state["gripper.pos"])

            try:
                async for _ in pace(fps):
                    iterations += 1

                    if time.monotonic() - t0 > reset_timeout_s:
                        reason = "timeout"
                        break

                    cur = {k: float(latest_state[k]) for k in ARM_POS_KEYS}
                    final_state = cur

                    diff = release_target - commanded_gripper
                    if abs(diff) <= reset_max_step:
                        commanded_gripper = release_target
                    elif diff > 0:
                        commanded_gripper = commanded_gripper + reset_max_step
                    else:
                        commanded_gripper = commanded_gripper - reset_max_step

                    # Mirror everything else from the latest observation
                    # so the arm holds its current pose while the gripper
                    # opens. Only `gripper.pos` diverges from `cur`.
                    next_action: dict[str, float] = {
                        k: cur[k] for k in ARM_POS_KEYS
                    }
                    next_action["gripper.pos"] = commanded_gripper
                    next_action["slider.vel"] = 0.0

                    op.send_action(
                        next_action,
                        timestamp_us=int(time.time() * 1_000_000),
                        in_reply_to_ts_us=latest_obs_ts_us,
                    )

                    gripper_remaining = abs(release_target - cur["gripper.pos"])
                    if gripper_remaining <= reset_tolerance:
                        reached = True
                        reason = "reached"
                        print(
                            f"[move-to-op] release_gripper reached at iter "
                            f"{iterations}: gripper.pos={cur['gripper.pos']:.2f}"
                        )
                        break

                    if iterations % fps == 0:
                        print(
                            f"[move-to-op] release tick {iterations:>4}: "
                            f"gripper.pos={cur['gripper.pos']:.2f} "
                            f"cmd={commanded_gripper:.2f} "
                            f"remaining={gripper_remaining:.2f}"
                        )
            finally:
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
                f"[move-to-op] release_gripper done: reason={reason} "
                f"reached={reached} iterations={iterations} "
                f"elapsed={elapsed_s:.2f}s"
            )

            return json.dumps({
                "target": {"gripper.pos": release_target},
                "final": final_state,
                "reached": reached,
                "reason": reason,
                "iterations": iterations,
                "elapsed_s": elapsed_s,
            })

    op.register_rpc_method("move_to", move_to)
    op.register_rpc_method("reset_pose", reset_pose)
    op.register_rpc_method("release_gripper", release_gripper)
    op.on_operator_joined(lambda i: print(f"[move-to-op] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[move-to-op] operator left: {i}"))

    print(f"[move-to-op] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()
    print(
        f"[move-to-op] connected as '{me}'; "
        "awaiting move_to/reset_pose/release_gripper RPCs"
    )

    try:
        # No tick loop. Control loops run only inside an active
        # move_to / reset_pose RPC; otherwise we park.
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
