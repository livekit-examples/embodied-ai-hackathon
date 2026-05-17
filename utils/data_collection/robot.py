"""Robot side: drives the physical leslider behind a Portal `Robot`.

Hardware setup, calibration, and dict reshaping live in ``utils/leslider.py``
so this file is the Portal flow alone. Each tick:

  1. ``follower.get_observation()`` produces state + camera frames.
  2. ``robot.send_state(...)`` and ``robot.send_video_frame(...)`` for each
     camera, all sharing one timestamp so the operator's synchronizer can
     match them into one observation.
  3. Apply the latest action received via ``on_action`` (the gate has
     already decided it should reach us based on ``active_operator``).
  4. Sleep to the next tick.

Stuck-pose safety: when the active operator drops out of the room, we
clear the cached latest action and the active-operator pointer. By
default Portal pins the pointer across disconnect (so a same-identity
reconnect resumes control), which is the wrong default for a physical
arm: it would hold the last commanded pose indefinitely. Explicit
clear here flips that policy.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time

from livekit.portal import Action, Robot, RobotConfig

from _common import env_int, load_env, mint_token, pace, required_env
from utils.leslider import CAMERAS, build_follower, split_state_frames

IDENTITY = "robot"
CONFIG_PATH = pathlib.Path(__file__).parent / "portal.yaml"

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)

    cfg = RobotConfig.from_yaml_file(CONFIG_PATH, room)
    fps = env_int("PORTAL_FPS", 30)

    follower = build_follower(fps)
    robot = Robot(cfg)

    latest_action: dict[str, float] = {}

    def on_action(action: Action) -> None:
        nonlocal latest_action
        latest_action = {k: float(v) for k, v in action.values.items()}

    def on_operator_left(identity: str) -> None:
        # If the active operator drops out, stop applying their last
        # cached action. Portal does not auto-clear the pointer on
        # disconnect; we do it explicitly so the arm idles instead of
        # holding a half-finished trajectory.
        nonlocal latest_action
        if identity == robot.active_operator():
            print(f"[robot] active operator '{identity}' left; releasing")
            latest_action = {}
            asyncio.create_task(robot.set_active_operator(None))

    robot.on_action(on_action)
    robot.on_operator_left(on_operator_left)
    robot.on_active_operator_changed(
        lambda i: print(f"[robot] active operator now: {i}")
    )

    follower.connect()
    print(f"[robot] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await robot.connect(url, token)
    print(f"[robot] connected; cameras={list(CAMERAS)}; streaming at {fps} fps; ctrl-c to stop")

    try:
        async for _ in pace(fps):
            ts_us = int(time.time() * 1_000_000)

            state, frames = split_state_frames(follower.get_observation())
            robot.send_state(state, timestamp_us=ts_us)
            for name, frame in frames.items():
                robot.send_video_frame(name, frame, timestamp_us=ts_us)

            if latest_action:
                follower.send_action(latest_action)

    except KeyboardInterrupt:
        print("\n[robot] stopping ...")
    finally:
        print("[robot] disconnecting...")
        try:
            await robot.disconnect()
        finally:
            robot.close()
            follower.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
