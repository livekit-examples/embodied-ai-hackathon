"""Drive the remote leslider with one of N trained ACT checkpoints.

A single Portal Operator that pre-loads multiple ACT policies (by
default ``pick_up`` and ``put_down``, trained via the
``utils/data_collection`` pipeline) and exposes an RPC

    run_policy(policy: "pick_up" | "put_down" | ...)

that:

  1. Resets the chosen policy's internal action chunk / temporal
     ensembler.
  2. Claims active-operator control of the robot.
  3. Switches the continuous tick loop to drive that policy.

The robot side runs the shared ``portal.yaml`` at the repo root, which
publishes *two* cameras (``arm_camera`` + ``overhead_camera``). The ACT
checkpoints trained via ``utils/data_collection`` typically only
consume one camera (its ``portal.yaml`` declares ``arm_camera`` only).
We filter the obs frames per-tick down to whatever each policy
declares in ``policy.config.input_features`` — extra cameras in the
obs are simply ignored.

State/action wire shape (7-dim, in the SO-101 follower's native order
from the shared ``portal.yaml``):

  [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
   wrist_flex.pos,   wrist_roll.pos,    gripper.pos,
   slider.vel]

ACT inference internals (chunk + temporal ensemble) follow
``utils/data_collection/policies/act/inference.py`` — referenced for
the policy/preprocessor/postprocessor wiring.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import DataProcessorPipeline

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    RpcError,
    RpcInvocationData,
    frame_bytes_to_numpy_rgb,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _common import env_int, env_str, load_env, mint_token, pace, required_env  # noqa: E402

IDENTITY = "run-policy-operator"


def _resolve_config_path() -> Path:
    override = os.environ.get("PORTAL_CONFIG")
    if override:
        return Path(override)
    for candidate in (ROOT / "portal.yaml", ROOT.parent.parent / "portal.yaml"):
        if candidate.exists():
            return candidate
    return ROOT.parent.parent / "portal.yaml"


CONFIG_PATH = _resolve_config_path()

# 7-dim wire schema from the shared portal.yaml. Action mirrors state.
STATE_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
    "slider.vel",
)
ACTION_KEYS = STATE_KEYS

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _policy_cameras(policy: ACTPolicy) -> tuple[str, ...]:
    """Cameras the loaded checkpoint expects, in declaration order, pulled
    from ``input_features``. Falls back to the single ``arm_camera`` if
    the checkpoint declares none (matches the data_collection default)."""
    keys = getattr(policy.config, "input_features", {}) or {}
    cams = tuple(
        k.removeprefix("observation.images.")
        for k in keys
        if k.startswith("observation.images.")
    )
    return cams or ("arm_camera",)


def _build_batch(
    state_typed: dict,
    frames: dict,
    policy_cameras: tuple[str, ...],
    task_prompt: str,
) -> dict:
    """Build a lerobot-style batch from a state dict + camera frames.

    Only `policy_cameras` are passed through; any extra cameras on the
    obs are dropped here so a two-camera robot can feed a one-camera
    policy."""
    state = torch.tensor(
        [float(state_typed[k]) for k in STATE_KEYS], dtype=torch.float32
    )
    batch: dict = {"observation.state": state}
    for cam in policy_cameras:
        img = frames.get(cam)
        if img is None:
            continue
        t = torch.from_numpy(np.ascontiguousarray(img))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        batch[f"observation.images.{cam}"] = t.contiguous()
    batch["task"] = task_prompt
    return batch


def _action_dict_from_tensor(action_tensor: torch.Tensor) -> dict[str, float]:
    a = action_tensor.detach().to("cpu").squeeze().tolist()
    return {k: float(a[i]) for i, k in enumerate(ACTION_KEYS)}


class LoadedPolicy:
    """Bundle of (policy, preprocessor, postprocessor, cameras) for one
    checkpoint. Owns its own reset() since ACT keeps internal chunk +
    temporal-ensemble state across calls."""

    def __init__(
        self,
        name: str,
        checkpoint_dir: Path,
        device: str,
        *,
        temporal_ensemble: bool,
        temporal_ensemble_coeff: float,
    ) -> None:
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.policy = ACTPolicy.from_pretrained(str(checkpoint_dir))
        self.policy.to(device)
        self.policy.eval()
        self.cameras = _policy_cameras(self.policy)

        if temporal_ensemble:
            self.policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff, self.policy.config.chunk_size
            )
        else:
            self.policy.config.temporal_ensemble_coeff = None

        self.preprocessor = DataProcessorPipeline.from_pretrained(
            str(checkpoint_dir),
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            str(checkpoint_dir),
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    def reset(self) -> None:
        self.policy.reset()


def _parse_policy_paths(args_pairs: list[str]) -> dict[str, Path]:
    """Parse ``name=path`` pairs into a mapping. Falls back to env vars
    ``PICK_UP_CHECKPOINT`` and ``PUT_DOWN_CHECKPOINT`` when no CLI pairs
    are given."""
    out: dict[str, Path] = {}
    if args_pairs:
        for p in args_pairs:
            if "=" not in p:
                raise SystemExit(
                    f"--policy expects name=path, got: {p!r}"
                )
            name, _, path = p.partition("=")
            name = name.strip()
            if not name or not path.strip():
                raise SystemExit(f"--policy expects name=path, got: {p!r}")
            out[name] = Path(path).expanduser().resolve()
        return out

    defaults = (
        ("pick_up", os.environ.get("PICK_UP_CHECKPOINT")),
        ("put_down", os.environ.get("PUT_DOWN_CHECKPOINT")),
    )
    for name, path in defaults:
        if path:
            out[name] = Path(path).expanduser().resolve()
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained ACT policies over LiveKit Portal, "
        "switchable via the run_policy(policy) RPC.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register a named policy checkpoint. Repeatable. "
             "Defaults to PICK_UP_CHECKPOINT / PUT_DOWN_CHECKPOINT env vars.",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        dest="temporal_ensemble",
        action="store_false",
        help="Disable temporal ensembling for all policies.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="Decay coefficient for temporal ensemble (default 0.01).",
    )
    parser.set_defaults(temporal_ensemble=True)
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    policy_paths = _parse_policy_paths(args.policy)
    if not policy_paths:
        raise SystemExit(
            "no policies registered; pass --policy name=path or set "
            "PICK_UP_CHECKPOINT / PUT_DOWN_CHECKPOINT"
        )

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)
    fps = env_int("PORTAL_FPS", 30)
    device = _select_device()
    task_prompt = env_str("LESLIDER_TASK", "Pick the blue cube up and hold it")

    print(f"[run-policy-operator] device={device}")
    for name, path in policy_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")
        print(f"[run-policy-operator]   policy '{name}' <- {path}")

    policies: dict[str, LoadedPolicy] = {}
    for name, path in policy_paths.items():
        print(f"[run-policy-operator] loading '{name}' ...")
        policies[name] = LoadedPolicy(
            name=name,
            checkpoint_dir=path,
            device=device,
            temporal_ensemble=args.temporal_ensemble,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
        print(
            f"[run-policy-operator]   '{name}' cameras={policies[name].cameras} "
            f"chunk={policies[name].policy.config.chunk_size}"
        )

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    latest_obs: Optional[Observation] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs

    op.on_observation(on_observation)
    op.on_active_operator_changed(
        lambda i: print(f"[run-policy-operator] active operator now: {i}")
    )

    # Currently-selected policy, or None to park. Single writer (the RPC
    # handler runs on the asyncio loop), single reader (the tick loop on
    # the same loop), so no lock needed on CPython.
    current: Optional[LoadedPolicy] = None

    async def run_policy_rpc(data: RpcInvocationData) -> str:
        nonlocal current
        payload = (data.payload or "").strip()
        name = payload
        if payload.startswith("{"):
            try:
                name = str(json.loads(payload).get("policy", "")).strip()
            except json.JSONDecodeError:
                raise RpcError.Error(
                    code=1400,
                    message="payload was not valid JSON; pass a bare policy "
                            "name string instead",
                    data=None,
                )
        if not name:
            raise RpcError.Error(
                code=1400,
                message=f"empty policy name; available: {sorted(policies)}",
                data=None,
            )
        if name not in policies:
            raise RpcError.Error(
                code=1404,
                message=f"unknown policy '{name}'; available: {sorted(policies)}",
                data=None,
            )

        chosen = policies[name]
        chosen.reset()
        current = chosen

        me_id = op.local_identity()
        await op.set_active_operator(me_id)
        print(
            f"[run-policy-operator] run_policy('{name}') -> "
            f"active operator -> {me_id} cameras={chosen.cameras}"
        )
        return json.dumps({
            "policy": name,
            "cameras": list(chosen.cameras),
            "active_operator": me_id,
        })

    op.register_rpc_method("run_policy", run_policy_rpc)

    print(
        f"[run-policy-operator] connecting to {url} as '{IDENTITY}' "
        f"in room '{room}' ..."
    )
    await op.connect(url, token)
    me = op.local_identity()
    print(
        f"[run-policy-operator] '{IDENTITY}' in '{room}' @ {fps} fps as '{me}' "
        f"(call run_policy({{pick_up|put_down}}) to engage)"
    )

    empty_obs_count = 0
    fwd_pass_count = 0
    fwd_pass_ms_sum = 0.0
    FWD_PASS_THRESHOLD_MS = 5.0
    tick_count = 0
    last_logged_policy: Optional[str] = None

    try:
        async for _ in pace(fps):
            obs = latest_obs
            if obs is None:
                empty_obs_count += 1
                if empty_obs_count in (30, 150, 600):
                    print(
                        f"[run-policy-operator] no observations for "
                        f"{empty_obs_count // 30}s, is robot.py running and "
                        "connected to the same room?"
                    )
                continue
            empty_obs_count = 0

            active = current
            if active is None:
                # No policy selected yet — park. The teleop side or any
                # operator can call run_policy() to engage.
                continue

            if last_logged_policy != active.name:
                last_logged_policy = active.name
                print(
                    f"[run-policy-operator] driving with '{active.name}' "
                    f"cameras={active.cameras}"
                )

            state_typed = dict(obs.state)
            frames: dict[str, np.ndarray] = {}
            for cam in active.cameras:
                vf = obs.frames.get(cam)
                if vf is not None:
                    frames[cam] = frame_bytes_to_numpy_rgb(
                        vf.data, vf.width, vf.height
                    )

            missing = [c for c in active.cameras if c not in frames]
            if missing:
                if tick_count == 0:
                    print(
                        f"[run-policy-operator] waiting for cameras "
                        f"{missing} required by '{active.name}' ..."
                    )
                continue

            batch = _build_batch(state_typed, frames, active.cameras, task_prompt)
            try:
                processed = active.preprocessor(batch)
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action_t = active.policy.select_action(processed)
                t_post = time.perf_counter()
                action_t = active.postprocessor(action_t)
            except Exception as e:
                print(f"[run-policy-operator] inference error: {e}")
                continue

            inf_ms = (t_post - t_inf) * 1000
            tick_count += 1
            if inf_ms > FWD_PASS_THRESHOLD_MS:
                fwd_pass_count += 1
                if fwd_pass_count > 1:
                    fwd_pass_ms_sum += inf_ms

            if tick_count % fps == 0:
                budget = 1000 / fps
                steady_n = max(fwd_pass_count - 1, 0)
                if steady_n > 0:
                    avg = fwd_pass_ms_sum / steady_n
                    marker = "OK" if avg < budget else "OVER"
                    print(
                        f"[run-policy-operator] tick {tick_count:>5} "
                        f"({active.name}): forward avg={avg:5.1f}ms "
                        f"(n={steady_n}) budget={budget:.1f}ms [{marker}]"
                    )

            action = _action_dict_from_tensor(action_t)
            op.send_action(
                action,
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us,
            )

    except KeyboardInterrupt:
        print("\n[run-policy-operator] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    asyncio.run(main())
