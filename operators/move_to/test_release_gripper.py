"""Quick CLI to exercise the `release_gripper` RPC on `move_to_operator.py`.

Joins the same LiveKit room as a separate operator participant, waits
for ``move_to_operator`` to be present, calls ``release_gripper()`` with
no payload, prints the result, and disconnects.

Usage::

    cp .env.example .env            # set LIVEKIT_*
    uv run move_to_operator.py      # in one shell
    uv run test_release_gripper.py  # in another
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from livekit.portal import Operator, OperatorConfig, PortalError

from _common import load_env, mint_token, required_env

IDENTITY = "release-gripper-tester"
DESTINATION = "move-to-operator"
METHOD = "release_gripper"
WAIT_S = 10.0
TIMEOUT_S = 30.0
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"


async def _wait_for_operator(op: Operator, identity: str, timeout: float) -> None:
    """Block until the operator with `identity` shows up, or raise TimeoutError.

    `on_operator_joined` only fires for *new* joins after registration; it
    does not replay the roster. So we both subscribe to the callback (in
    case the target hasn't connected yet) *and* poll `op.operators()`
    (the live roster, populated as peers' `lk.portal.role` attributes
    propagate). Either path can set the event.
    """
    found = asyncio.Event()

    def _on_joined(other: str) -> None:
        if other == identity:
            found.set()

    op.on_operator_joined(_on_joined)

    async def _poll() -> None:
        while not found.is_set():
            if identity in op.operators():
                found.set()
                return
            await asyncio.sleep(0.1)

    poll_task = asyncio.create_task(_poll())
    try:
        await asyncio.wait_for(found.wait(), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            f"'{identity}' not in room after {timeout:.0f}s. "
            f"Is move_to_operator.py running with the same LIVEKIT_ROOM?"
        ) from e
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass


async def main() -> int:
    load_env()

    url = required_env("LIVEKIT_URL")
    room_name = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room_name)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room_name)
    op = Operator(cfg)

    print(f"[test] connecting to {url} as '{IDENTITY}' in room '{room_name}' ...")
    await op.connect(url, token)
    print(f"[test] connected; waiting up to {WAIT_S:.0f}s for '{DESTINATION}' ...")

    try:
        await _wait_for_operator(op, DESTINATION, WAIT_S)
        print(f"[test] calling {METHOD}() on '{DESTINATION}' ...")
        try:
            response = await op.perform_rpc(
                METHOD,
                destination=DESTINATION,
                response_timeout_ms=int(TIMEOUT_S * 1000),
            )
        except PortalError as e:
            print(f"[test] RPC error: {e}")
            return 2

        try:
            parsed = json.loads(response)
            print(f"[test] result: {json.dumps(parsed, indent=2)}")
        except (json.JSONDecodeError, TypeError):
            print(f"[test] result: {response}")
        return 0
    except TimeoutError as e:
        print(f"[test] {e}")
        return 3
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
