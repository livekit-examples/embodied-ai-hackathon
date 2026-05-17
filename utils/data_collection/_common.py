"""Shared helpers for robot.py / teleoperator.py / policies/<algo>/inference.py.

Reads LiveKit API credentials from .env, mints JWTs with low-latency
playout bounds, and provides a single FPS pacer. Kept tiny on purpose;
hardware and rerun-specific helpers live in ``utils/``.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import pathlib
import time
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.room import RoomConfiguration


def load_env(search_from: Optional[pathlib.Path] = None) -> None:
    """Load `.env` then `.env.local` from the script dir or its parent."""
    start = search_from or pathlib.Path(__file__).parent
    for d in (start, start.parent, pathlib.Path.cwd()):
        env = d / ".env"
        if env.exists():
            load_dotenv(env, override=False)
        env_local = d / ".env.local"
        if env_local.exists():
            load_dotenv(env_local, override=True)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return value


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def env_camera_id(name: str, default: str) -> int | str:
    """Camera identifier, int index on macOS, /dev/videoN string on Linux."""
    raw = os.environ.get(name, default)
    return int(raw) if raw.isdigit() else raw


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    """Mint a LiveKit JWT scoped to `room` with low-latency playout bounds.

    `can_update_own_metadata` is required so Portal can publish its
    `lk.portal.role` attribute on connect (used for peer discovery).
    """
    key = required_env("LIVEKIT_API_KEY")
    secret = required_env("LIVEKIT_API_SECRET")
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    room_cfg = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(room_cfg)
        .with_ttl(datetime.timedelta(hours=ttl_hours))
        .to_jwt()
    )


async def pace(fps: int) -> AsyncIterator[int]:
    """Async tick generator at `fps`. Yields the index, then awaits.

    Async (not `time.sleep`) so Portal callbacks scheduled via
    `call_soon_threadsafe` actually run between ticks. Drift is corrected
    by snapping `next_tick` forward when the loop is over budget, same
    behaviour as the built-in basic example.

    Use as `async for tick in pace(30): ...`.
    """
    interval = 1.0 / fps
    next_tick = time.perf_counter()
    i = 0
    while True:
        yield i
        i += 1
        next_tick += interval
        now = time.perf_counter()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_tick = now
