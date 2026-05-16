"""Small shared runtime helpers for the standalone robot process."""
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
    for directory in (start, start.parent, pathlib.Path.cwd()):
        env = directory / ".env"
        if env.exists():
            load_dotenv(env, override=False)
        env_local = directory / ".env.local"
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


def env_camera_id(name: str, default: str) -> int | str:
    """Camera identifier, int index on macOS, /dev/videoN string on Linux."""
    raw = os.environ.get(name, default)
    return int(raw) if raw.isdigit() else raw


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    """Mint a LiveKit JWT scoped to `room` for the Portal robot."""
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
    """Async tick generator at `fps`."""
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
