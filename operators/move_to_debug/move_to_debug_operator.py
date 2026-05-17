"""Operator side: perception-only `move_to` debug observer.

A read-only sibling of ``move_to_operator``. Joins the leslider room
as an Operator peer, caches the latest ``overhead_camera`` frame, and
exposes a ``move_to(description)`` RPC that runs a tracking loop
**without taking control of the robot**:

  1. Asks Moondream's cloud ``point`` endpoint to find the described
     object in the current overhead frame, prints its pixel-Y.
  2. Runs a tracking loop until ``DEBUG_TRACK_TIMEOUT_S`` elapses:
     - On every tick, runs the AprilTag detector on the latest frame
       and prints the slider tag's pixel-Y.
     - Periodically (default once per second, see
       ``DEBUG_OBJECT_DETECT_INTERVAL_S``), re-runs Moondream's
       ``point`` endpoint to relocate the requested object and prints
       its pixel-Y.
  3. Returns a JSON summary with the last detected positions and tick
     counts.

Since the camera is reoriented so the slider's travel axis runs
up/down in the image, only the Y coordinate is reported here — that
is what the real ``move_to_operator`` uses to drive the slider, so
matching its observability one-to-one keeps calibration honest.

Crucially, this operator never calls ``set_active_operator`` and never
calls ``send_action``. It is inert on the wire. Use it to verify the
perception layer (AprilTag visibility, Moondream detection quality,
frame freshness) and to compare the tag's Y against the target
object's Y before trusting the real controller, without putting the
slider in motion.

Mirrors ``move_to_operator``'s RPC name (``move_to``) so the existing
``test_move_to.py`` works unchanged via ``--destination
move-to-debug-operator``.

Environment knobs (in addition to LIVEKIT_* and MOONDREAM_API_KEY):

* ``SLIDER_APRILTAG_FAMILY``  default ``tag16h5``
* ``SLIDER_APRILTAG_ID``      optional integer; if set, only that tag
                              ID is accepted as the slider fiducial.
* ``DEBUG_TRACK_TIMEOUT_S``   loop duration in seconds; default ``20``.
* ``DEBUG_TAG_PRINT_INTERVAL_S`` minimum seconds between consecutive
                              tag detection prints. The detector still
                              runs every tick (so ``last_tag`` stays
                              fresh) but prints are gated by wall time
                              to keep the console readable. Default
                              ``1.0``.
* ``DEBUG_OBJECT_DETECT_INTERVAL_S`` seconds between consecutive
                              Moondream ``point`` calls. Default
                              ``1.0``.

Usage::

    cp .env.example .env  # set LIVEKIT_* and MOONDREAM_API_KEY
    uv sync
    uv run move_to_debug_operator.py

Then from any peer in the same room::

    result = await op.perform_rpc(
        "move_to", "blue cube", destination="move-to-debug-operator"
    )
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
    env_float,
    env_int,
    env_str,
    load_env,
    mint_token,
    pace,
    required_env,
)

IDENTITY = "move-to-debug-operator"
CAMERA = "overhead_camera"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

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

    # Read once at startup; tweak via .env and restart to change.
    track_timeout_s = env_int("DEBUG_TRACK_TIMEOUT_S", 20)
    tag_print_interval_s = max(0.0, env_float("DEBUG_TAG_PRINT_INTERVAL_S", 1.0))
    object_interval_s = max(0.05, env_float("DEBUG_OBJECT_DETECT_INTERVAL_S", 1.0))
    tag_family = env_str("SLIDER_APRILTAG_FAMILY", "tag16h5")
    raw_tag_id = os.environ.get("SLIDER_APRILTAG_ID", "").strip()
    want_tag_id: Optional[int] = int(raw_tag_id) if raw_tag_id else None

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    # Built once at startup so the per-RPC path is just an HTTP call,
    # not an SDK init. Each Moondream call is dispatched via
    # `asyncio.to_thread` so it can't block the AprilTag tick loop.
    model = md.vl(api_key=api_key)

    # AprilTag detector reused across RPCs and loop iterations. Same
    # `quad_decimate=1.0` tuning as `move_to_operator` — the slider tag
    # is small in the overhead frame and we'd rather pay CPU than miss
    # detections.
    tag_detector = Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
    )
    print(
        f"[move-to-debug] apriltag detector ready: family={tag_family!r} "
        f"want_id={want_tag_id}"
    )
    print(
        f"[move-to-debug] track_timeout={track_timeout_s}s, "
        f"tag_print_interval={tag_print_interval_s:.2f}s, "
        f"object_interval={object_interval_s:.2f}s"
    )

    # Updated on every synchronized obs. One-writer / one-reader: the
    # Portal callback (already hopped onto the asyncio loop) writes,
    # the RPC handler reads. No lock needed for atomic rebinds on
    # CPython. We don't track `latest_state` — this operator never
    # publishes actions, so robot state is irrelevant here.
    latest_overhead: Optional[np.ndarray] = None
    latest_obs_ts_us: int = 0

    def on_observation(obs: Observation) -> None:
        nonlocal latest_overhead, latest_obs_ts_us
        f = obs.frames.get(CAMERA)
        if f is not None:
            latest_overhead = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
        latest_obs_ts_us = obs.timestamp_us

    op.on_observation(on_observation)

    async def move_to(data: RpcInvocationData) -> str:
        # Accept either a bare string ("blue cube") or a JSON object
        # ({"description": "blue cube"}). Same shape as
        # `move_to_operator` so the existing test scripts work.
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

        h, w, _ = frame.shape
        image = Image.fromarray(frame)
        try:
            initial = await asyncio.to_thread(model.point, image, description)
        except Exception as e:  # noqa: BLE001
            logger.exception("moondream point() failed")
            raise RpcError.Error(
                code=1502,
                message=f"moondream error: {type(e).__name__}: {e}",
                data=None,
            )

        points = initial.get("points") or []
        if not points:
            raise RpcError.Error(
                code=1404,
                message=f"'{description}' not found in current frame",
                data=None,
            )

        p0 = points[0]
        target_yn = float(p0["y"])
        target_y_px = target_yn * h
        print(
            f"[move-to-debug] move_to('{description}') -> "
            f"target yn={target_yn:.3f} y_px={target_y_px:.1f} frame={w}x{h}"
        )

        # Latest detections so the JSON summary can echo something
        # useful. None until the first successful detection. Only Y
        # is tracked — that's the slider's travel axis after the
        # camera reorient, so X carries no actionable info here.
        last_tag_y_px: Optional[float] = None
        last_object_y_px: Optional[float] = target_y_px
        last_object_yn: Optional[float] = target_yn

        tag_iters = 0
        tag_hits = 0
        tag_miss = 0
        object_iters = 1  # initial detection counts
        object_hits = 1
        object_miss = 0

        t0 = time.monotonic()
        deadline = t0 + track_timeout_s

        # Concurrency: one task ticks the AprilTag detector at fps
        # (cheap, CPU-local), another paces Moondream over the network
        # (~1 Hz default). They share `latest_overhead` and don't
        # touch each other's counters except via nonlocal rebinds,
        # which are atomic on CPython.
        async def tag_loop() -> None:
            nonlocal last_tag_y_px, tag_iters, tag_hits, tag_miss
            # Wall-clock gate: detector still runs every tick to keep
            # `last_tag_y_px` fresh, but console prints are rate-limited
            # so the log stays readable at high fps.
            last_print_t = 0.0
            async for _ in pace(fps):
                now = time.monotonic()
                if now >= deadline:
                    return
                tag_iters += 1
                cur = latest_overhead
                if cur is None:
                    tag_miss += 1
                    continue
                tag = _detect_slider_tag(tag_detector, cur, want_tag_id)
                should_print = (now - last_print_t) >= tag_print_interval_s
                if tag is None:
                    tag_miss += 1
                    if should_print:
                        last_print_t = now
                        print(
                            f"[move-to-debug] tag tick={tag_iters:>4}: "
                            f"no detection"
                        )
                    continue
                tag_hits += 1
                last_tag_y_px = tag[1]
                if should_print:
                    last_print_t = now
                    print(
                        f"[move-to-debug] tag tick={tag_iters:>4}: "
                        f"y={tag[1]:.1f}"
                    )

        async def object_loop() -> None:
            nonlocal last_object_y_px, last_object_yn
            nonlocal object_iters, object_hits, object_miss
            # Sleep first; the initial detection above already covers
            # t=0, so re-detecting immediately would just duplicate it.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(object_interval_s, remaining))
                if time.monotonic() >= deadline:
                    return
                object_iters += 1
                cur = latest_overhead
                if cur is None:
                    object_miss += 1
                    print(
                        f"[move-to-debug] object iter={object_iters}: "
                        "no frame"
                    )
                    continue
                cur_image = Image.fromarray(cur)
                try:
                    result = await asyncio.to_thread(
                        model.point, cur_image, description
                    )
                except Exception as e:  # noqa: BLE001
                    object_miss += 1
                    print(
                        f"[move-to-debug] object iter={object_iters}: "
                        f"moondream error: {type(e).__name__}: {e}"
                    )
                    continue
                pts = result.get("points") or []
                if not pts:
                    object_miss += 1
                    print(
                        f"[move-to-debug] object iter={object_iters}: "
                        f"'{description}' not found"
                    )
                    continue
                object_hits += 1
                p = pts[0]
                yn = float(p["y"])
                ch, _, _ = cur.shape
                y_px = yn * ch
                last_object_yn = yn
                last_object_y_px = y_px
                print(
                    f"[move-to-debug] object iter={object_iters}: "
                    f"'{description}' yn={yn:.3f} y_px={y_px:.1f}"
                )

        try:
            await asyncio.gather(tag_loop(), object_loop())
        except asyncio.CancelledError:
            # Bubble cancellation up so the RPC layer can record it.
            # No control or action state to clean up — this operator
            # never claims either.
            raise

        elapsed_s = time.monotonic() - t0
        print(
            f"[move-to-debug] move_to done: elapsed={elapsed_s:.2f}s "
            f"tag_iters={tag_iters} tag_hits={tag_hits} tag_miss={tag_miss} "
            f"object_iters={object_iters} object_hits={object_hits} "
            f"object_miss={object_miss}"
        )

        return json.dumps({
            "description": description,
            "frame": {"width": w, "height": h},
            "initial_target": {
                "y": target_yn,
                "y_px": target_y_px,
            },
            "last_object": (
                {"y": last_object_yn, "y_px": last_object_y_px}
                if last_object_yn is not None
                else None
            ),
            "last_tag": (
                {"y_px": last_tag_y_px}
                if last_tag_y_px is not None
                else None
            ),
            "tag_iters": tag_iters,
            "tag_hits": tag_hits,
            "tag_miss": tag_miss,
            "object_iters": object_iters,
            "object_hits": object_hits,
            "object_miss": object_miss,
            "elapsed_s": elapsed_s,
            "took_control": False,
        })

    op.register_rpc_method("move_to", move_to)
    op.on_operator_joined(lambda i: print(f"[move-to-debug] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[move-to-debug] operator left: {i}"))

    print(f"[move-to-debug] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()
    print(
        f"[move-to-debug] connected as '{me}'; "
        "awaiting move_to RPCs (read-only; never claims active operator)"
    )

    try:
        # No tick loop. Tracking loops run only inside an active
        # move_to RPC; otherwise we park.
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n[move-to-debug] stopping ...")
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
