"""Quick CLI to exercise the `reset_pose` RPC on `move_to_operator.py`.

Joins the same LiveKit room as a *separate* operator participant
(different identity), waits for ``move_to_operator`` to be present,
calls ``reset_pose()``, prints the result, and disconnects.

Same shape as ``test_move_to.py``: pure RPC client, no observation
subscription, no actions sent. The operator's ``reset_pose`` ignores
its payload, so this script sends an empty string by default; pass
``--json`` to send ``{}`` instead if you want to exercise that path.

Usage::

    cp .env.example .env       # set LIVEKIT_*
    uv run move_to_operator.py # in one shell
    uv run test_reset_pose.py  # in another

Flags::

    --destination   Operator identity to address. Defaults to
                    ``move-to-operator`` (matches that script's IDENTITY).
    --method        RPC method name. Defaults to ``reset_pose``.
    --identity      Our own participant identity in the room. Defaults to
                    ``reset-pose-tester``.
    --room          Override LIVEKIT_ROOM from .env.
    --timeout       Per-call response timeout in seconds (default 30).
                    The operator's reset loop has its own
                    ``RESET_POSE_TIMEOUT_S`` (default 10s); this RPC
                    timeout should comfortably exceed that so a slow
                    ramp doesn't time out client-side before the
                    operator finishes.
    --wait          How long to wait for the destination operator to
                    show up before invoking (default 10).
    --json          Send payload as ``{}`` instead of an empty string.
                    The operator ignores the payload; this is mainly
                    useful for testing the JSON parse path.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from livekit.portal import Operator, OperatorConfig, PortalError

from _common import load_env, mint_token, required_env

DEFAULT_IDENTITY = "reset-pose-tester"
DEFAULT_DESTINATION = "move-to-operator"
DEFAULT_METHOD = "reset_pose"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--destination", default=DEFAULT_DESTINATION, help="Operator identity to address")
    p.add_argument("--method", default=DEFAULT_METHOD, help="RPC method name")
    p.add_argument("--identity", default=DEFAULT_IDENTITY, help="Our participant identity")
    p.add_argument("--room", default=None, help="Room name (default: LIVEKIT_ROOM from .env)")
    p.add_argument("--timeout", type=float, default=30.0, help="RPC response timeout (s)")
    p.add_argument("--wait", type=float, default=10.0, help="Seconds to wait for the destination to appear")
    p.add_argument("--json", action="store_true", help="Send payload as an empty JSON object instead of an empty string")
    return p.parse_args()


async def _wait_for_operator(op: Operator, identity: str, timeout: float) -> None:
    """Block until the operator with `identity` shows up, or raise TimeoutError.

    `on_operator_joined` only fires for *new* joins after registration; it
    does not replay the roster. So we both subscribe to the callback (in
    case the target hasn't connected yet) *and* poll `op.operators()`
    (the live roster, populated as peers' `lk.portal.role` attributes
    propagate). Either path can set the event.
    """
    found = asyncio.Event()
    seen: set[str] = set()

    def _on_joined(other: str) -> None:
        seen.add(other)
        if other == identity:
            found.set()

    op.on_operator_joined(_on_joined)

    async def _poll() -> None:
        while not found.is_set():
            current = op.operators()
            seen.update(current)
            if identity in current:
                found.set()
                return
            await asyncio.sleep(0.1)

    poll_task = asyncio.create_task(_poll())
    try:
        await asyncio.wait_for(found.wait(), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            f"'{identity}' not in room after {timeout:.0f}s "
            f"(saw operators: {sorted(seen) or 'nobody'}). "
            f"Is move_to_operator.py running with the same LIVEKIT_ROOM?"
        ) from e
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass


async def main() -> int:
    args = _parse_args()
    load_env()

    url = required_env("LIVEKIT_URL")
    room_name = args.room or required_env("LIVEKIT_ROOM")
    token = mint_token(args.identity, room_name)

    payload = "{}" if args.json else ""

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room_name)
    op = Operator(cfg)

    print(f"[test] connecting to {url} as '{args.identity}' in room '{room_name}' ...")
    await op.connect(url, token)
    print(f"[test] connected; waiting up to {args.wait:.0f}s for '{args.destination}' ...")

    try:
        await _wait_for_operator(op, args.destination, args.wait)
        print(f"[test] calling {args.method}({payload!r}) on '{args.destination}' ...")
        try:
            response = await op.perform_rpc(
                args.method,
                payload=payload,
                destination=args.destination,
                response_timeout_ms=int(args.timeout * 1000),
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
