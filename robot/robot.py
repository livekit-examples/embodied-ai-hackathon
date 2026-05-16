"""Portal robot process for the physical leslider follower."""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from collections import defaultdict

from livekit.portal import Action, PortalError, Robot, RobotConfig

from _common import env_int, load_env, mint_token, pace, required_env
from utils.leslider import CAMERAS, build_follower, split_state_frames

IDENTITY = "robot"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)

    cfg = RobotConfig.from_yaml_file(CONFIG_PATH, room)
    fps = env_int("PORTAL_FPS", 30)

    # h264 tracks ride the WebRTC media path; mjpeg/png/raw ride a byte
    # stream and never show up as a video track in the room. Print both
    # so it's obvious which path is wired up after the YAML parse.
    logger.info(
        "[robot] portal.yaml parsed: webrtc(h264) tracks=%s, "
        "frame-video(mjpeg/png/raw) tracks=%s",
        list(cfg.video_tracks),
        [(s.name, str(s.codec)) for s in cfg.frame_video_tracks],
    )
    if not cfg.video_tracks and not cfg.frame_video_tracks:
        logger.warning("[robot] no video tracks declared in portal.yaml")

    follower = build_follower(fps)
    robot = Robot(cfg)
    latest_action: dict[str, float] = {}

    def on_action(action: Action) -> None:
        nonlocal latest_action
        latest_action = {key: float(value) for key, value in action.values.items()}

    def on_operator_left(identity: str) -> None:
        nonlocal latest_action
        if identity == robot.active_operator():
            logger.info("[robot] active operator '%s' left; releasing", identity)
            latest_action = {}
            asyncio.create_task(robot.set_active_operator(None))

    robot.on_action(on_action)
    robot.on_operator_left(on_operator_left)
    robot.on_active_operator_changed(
        lambda identity: logger.info("[robot] active operator now: %s", identity)
    )

    follower.connect()
    logger.info("[robot] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await robot.connect(url, token)
    logger.info(
        "[robot] connected; cameras=%s; streaming at %d fps; ctrl-c to stop",
        list(CAMERAS),
        fps,
    )

    # Per-tick counters for the diagnostic summary. These let us localize
    # a "no video in the browser" failure to one of:
    #   captured == 0          -> cv2/V4L2 isn't producing frames
    #   pushed  == 0 (cap > 0) -> split_state_frames is dropping them
    #   sent (metrics) == 0    -> portal isn't forwarding to libwebrtc
    #   send errors > 0        -> dimension/size mismatch on a frame
    captured: defaultdict[str, int] = defaultdict(int)
    pushed: defaultdict[str, int] = defaultdict(int)
    send_errors: defaultdict[str, int] = defaultdict(int)
    last_logged_frame_shape: dict[str, tuple[int, int]] = {}

    try:
        ticks = 0
        async for _ in pace(fps):
            ticks += 1
            ts_us = int(time.time() * 1_000_000)

            state, frames = split_state_frames(follower.get_observation())
            robot.send_state(state, timestamp_us=ts_us)

            for name in CAMERAS:
                frame = frames.get(name)
                if frame is None:
                    continue
                captured[name] += 1
                # One-shot shape log per camera so we can confirm dims
                # are even (h264 publisher rejects odd width/height).
                shape = (int(frame.shape[1]), int(frame.shape[0]))
                if last_logged_frame_shape.get(name) != shape:
                    last_logged_frame_shape[name] = shape
                    logger.info(
                        "[robot] '%s' first/changed frame shape: %dx%d, dtype=%s",
                        name,
                        shape[0],
                        shape[1],
                        getattr(frame, "dtype", "?"),
                    )
                try:
                    robot.send_video_frame(name, frame, timestamp_us=ts_us)
                    pushed[name] += 1
                except PortalError as e:
                    send_errors[name] += 1
                    if send_errors[name] <= 3 or send_errors[name] % fps == 0:
                        logger.warning(
                            "[robot] send_video_frame('%s') failed: %s", name, e
                        )

            if latest_action:
                follower.send_action(latest_action)

            # Once per second: dump capture/push/wire counters so the
            # symptom maps cleanly to a stage. `transport.frames_sent`
            # is bumped by the Rust core inside VideoPublisher::send_frame
            # right after capture_frame(), so it's the closest readout
            # to "did libwebrtc accept this frame".
            if ticks % fps == 0:
                m = robot.metrics().transport
                rows = []
                for name in CAMERAS:
                    rows.append(
                        f"{name}: captured={captured[name]} "
                        f"pushed={pushed[name]} "
                        f"wire_sent={m.frames_sent.get(name, 0)} "
                        f"errors={send_errors[name]}"
                    )
                logger.info("[robot] video stats (cumulative): %s", " | ".join(rows))

    except KeyboardInterrupt:
        logger.info("[robot] stopping ...")
    finally:
        logger.info("[robot] disconnecting...")
        try:
            await robot.disconnect()
        finally:
            robot.close()
            follower.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
