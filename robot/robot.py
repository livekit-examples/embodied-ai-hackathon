"""Portal robot process for the physical leslider follower."""
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
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "portal.yaml"

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
        latest_action = {key: float(value) for key, value in action.values.items()}

    def on_operator_left(identity: str) -> None:
        nonlocal latest_action
        if identity == robot.active_operator():
            print(f"[robot] active operator '{identity}' left; releasing")
            latest_action = {}
            asyncio.create_task(robot.set_active_operator(None))

    robot.on_action(on_action)
    robot.on_operator_left(on_operator_left)
    robot.on_active_operator_changed(
        lambda identity: print(f"[robot] active operator now: {identity}")
    )

    follower.connect()
    print(f"[robot] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await robot.connect(url, token)
    print(
        f"[robot] connected; cameras={list(CAMERAS)}; "
        f"streaming at {fps} fps; ctrl-c to stop"
    )

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
