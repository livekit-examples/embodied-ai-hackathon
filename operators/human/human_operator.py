"""Human teleop operator for the leslider rig.

Joins the LiveKit room as a Portal Operator peer. Each tick, reads one
action dict from the SO-101 leader (with built-in arrow-key slider
control) and forwards it to the wire. Mirrors observations into rerun
so the operator can see the remote cameras while flying.

This is the human-teleop slice of `portal-hitl` with the dataset
recorder removed — no episodes are collected here.

Hotkeys:
  c: cycle the active operator through ``[self, *remote_operators]``,
     wrapping back to self. With no remote operators connected, this
     just toggles claim/release for self.
  x: clean quit.

Slider drive lives on the leader's own listener: `←/→` hold to drive,
`↑/↓` to trim cruise speed, space to stop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
import time
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

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _common import env_int, load_env, mint_token, pace, required_env  # noqa: E402
from utils.hotkeys import Hotkeys  # noqa: E402
from utils.leslider import CAMERAS, build_leader  # noqa: E402
from utils.rerun_view import (  # noqa: E402
    build_blueprint,
    log_action,
    log_metrics,
    log_observation,
)

IDENTITY = "human-operator"
CONFIG_PATH = ROOT.parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


async def cycle_active_operator(op: Operator, me: str | None) -> None:
    """`c`: cycle active operator through ``[self, *others]`` and wrap
    back to self. With no remote operator connected the ring is just
    ``[self]``, so pressing `c` claims control for self — the common
    case during solo teleop, when the active operator is still ``None``
    and the robot is dropping our actions."""
    ring = [me, *op.operators()]
    current = op.active_operator()
    try:
        idx = ring.index(current)
    except ValueError:
        idx = -1
    nxt = ring[(idx + 1) % len(ring)]
    await op.set_active_operator(nxt)
    print(f"[human] active operator → {nxt}")


async def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    fps = env_int("PORTAL_FPS", 30)

    leader = build_leader()
    op = Operator(cfg)

    latest_obs: Optional[Observation] = None
    latest_state: dict = {}
    latest_frames: dict[str, np.ndarray] = {}
    latest_executed: Optional[Action] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs, latest_state
        latest_obs = obs
        latest_state = dict(obs.state)
        for cam in CAMERAS:
            f = obs.frames.get(cam)
            if f is not None:
                latest_frames[cam] = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)

    def on_action(action: Action) -> None:
        nonlocal latest_executed
        latest_executed = action

    op.on_observation(on_observation)
    op.on_action(on_action)
    op.on_active_operator_changed(
        lambda i: print(f"[human] active operator now: {i}")
    )
    op.on_operator_joined(lambda i: print(f"[human] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[human] operator left: {i}"))

    leader.connect()
    print(f"[human] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()

    print(f"[human] connected as '{me}' @ {fps} fps")
    print("[human] hotkeys: c=cycle operator  x=quit")
    print("[human] slider keys: ←/→ drive  ↑/↓ trim  space stop  (handled by leader)")

    rr.init(f"human-operator-{room}", spawn=True)
    rr.send_blueprint(build_blueprint())

    hotkeys = Hotkeys()
    hotkeys.start()
    quit_requested = False

    try:
        async for tick in pace(fps):
            for key in hotkeys.pop_pressed():
                if key == "c":
                    await cycle_active_operator(op, me)
                elif key == "x":
                    quit_requested = True
            if quit_requested:
                break

            # Drive the wire. Always send, even when not active; the
            # robot's gate drops mismatched-sender actions, and keeping
            # the leader in sync means takeover is instant.
            leader_action = leader.get_action()
            op.send_action(leader_action, timestamp_us=int(time.time() * 1_000_000))

            if tick % fps == 0:
                pairs = ", ".join(
                    f"{k}={float(v):+.2f}" for k, v in leader_action.items()
                )
                print(f"[human] leader action: {pairs}")

            log_observation(latest_obs, latest_state, latest_frames)
            log_action(latest_executed, me)
            if tick % fps == 0 and (m := op.metrics()) is not None:
                log_metrics(m)

    except KeyboardInterrupt:
        print("\n[human] stopping ...")
    finally:
        hotkeys.stop()
        try:
            await op.disconnect()
        finally:
            op.close()
            leader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
