"""Drive the remote leslider with one of N trained ACT checkpoints.

A single Portal Operator that pre-loads multiple ACT policies (by
default ``pick_up`` and ``put_down``, trained via the
``utils/data_collection`` pipeline) and exposes an RPC

    run_policy(policy: "pick_up" | "put_down" | ...)

that:

  1. Resets the chosen policy's internal action chunk / temporal
     ensembler (and the sparse-reward "done" counter, if a reward
     checkpoint is attached to the chosen policy).
  2. Claims active-operator control of the robot.
  3. Switches the continuous tick loop to drive that policy.
  4. Blocks the RPC reply until a "task done" signal fires. Two
     signal sources are supported, and any policy can use either or
     both:
       * a sparse-reward classifier (``utils/train_sparse_reward``)
         attached to the policy. Fires when ``P(done) >= threshold``
         for ``trigger_ticks`` consecutive ticks.
       * a fixed driving duration in seconds. Fires after that many
         seconds of driving, regardless of the reward signal.
     If both are configured, whichever fires first wins. On done,
     the tick loop parks, the active-operator slot is released, and
     the RPC returns success. Policies with neither signal source
     return immediately (``status: "no_terminator"``).

The robot side runs the shared ``portal.yaml`` at the repo root, which
publishes *two* cameras (``arm_camera`` + ``overhead_camera``). The ACT
checkpoints trained via ``utils/data_collection`` typically only
consume one camera (its ``portal.yaml`` declares ``arm_camera`` only).
We filter the obs frames per-tick down to whatever each policy
declares in ``policy.config.input_features``; extra cameras in the
obs are simply ignored.

State/action wire shape (7-dim, in the SO-101 follower's native order
from the shared ``portal.yaml``):

  [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
   wrist_flex.pos,   wrist_roll.pos,    gripper.pos,
   slider.vel]

ACT inference internals (chunk + temporal ensemble) follow
``utils/data_collection/policies/act/inference.py``, referenced for
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

# Sparse-reward classifier ships as a flat-script project under
# utils/train_sparse_reward (see its README). It has no install step,
# so we pull it in via sys.path on demand.
SPARSE_REWARD_DIR = ROOT.parent.parent / "utils" / "train_sparse_reward"


def _load_reward_scorer_cls():
    if not SPARSE_REWARD_DIR.is_dir():
        raise FileNotFoundError(
            f"sparse-reward source not found at {SPARSE_REWARD_DIR}; "
            "expected utils/train_sparse_reward/inference.py"
        )
    if str(SPARSE_REWARD_DIR) not in sys.path:
        sys.path.insert(0, str(SPARSE_REWARD_DIR))
    from inference import RewardScorer  # noqa: WPS433
    return RewardScorer


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
    """Bundle of (policy, preprocessor, postprocessor, cameras, optional
    sparse-reward scorer) for one checkpoint. Owns its own reset() since
    ACT keeps internal chunk + temporal-ensemble state across calls, and
    we use the same hook to reset the per-run reward counter."""

    def __init__(
        self,
        name: str,
        checkpoint_dir: Path,
        device: str,
        *,
        temporal_ensemble: bool,
        temporal_ensemble_coeff: float,
        reward_checkpoint: Optional[Path] = None,
        reward_camera: str = "arm_camera",
        duration_s: Optional[float] = None,
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

        # Optional sparse-reward classifier. Same single-step inputs as
        # ACT (one camera + 7-d state); see utils/train_sparse_reward.
        self.reward_checkpoint = reward_checkpoint
        self.reward_camera = reward_camera
        self.scorer = None
        if reward_checkpoint is not None:
            RewardScorer = _load_reward_scorer_cls()
            self.scorer = RewardScorer.from_checkpoint(
                reward_checkpoint, device=device
            )

        # Optional fixed-duration auto-stop. Independent of the reward
        # classifier; whichever terminator fires first wins.
        self.duration_s = duration_s

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


def _parse_reward_paths(args_pairs: list[str]) -> dict[str, Path]:
    """Same shape as `_parse_policy_paths`, but for sparse-reward
    checkpoint paths keyed by policy name. Falls back to env vars
    ``PICK_UP_REWARD`` / ``PUT_DOWN_REWARD`` when no CLI pairs are
    given. Policies without an entry here will not auto-stop and the
    RPC returns immediately for them."""
    out: dict[str, Path] = {}
    if args_pairs:
        for p in args_pairs:
            if "=" not in p:
                raise SystemExit(
                    f"--reward expects name=path, got: {p!r}"
                )
            name, _, path = p.partition("=")
            name = name.strip()
            if not name or not path.strip():
                raise SystemExit(f"--reward expects name=path, got: {p!r}")
            out[name] = Path(path).expanduser().resolve()
        return out

    defaults = (
        ("pick_up", os.environ.get("PICK_UP_REWARD")),
        ("put_down", os.environ.get("PUT_DOWN_REWARD")),
    )
    for name, path in defaults:
        if path:
            out[name] = Path(path).expanduser().resolve()
    return out


def _parse_duration_specs(args_pairs: list[str]) -> dict[str, float]:
    """Parse ``name=seconds`` pairs into per-policy fixed-duration auto-stops.

    A policy with a duration entry will auto-park after ``seconds`` of
    driving, even without a reward checkpoint. Combined with a reward
    checkpoint, whichever fires first wins. Falls back to env vars
    ``PICK_UP_DURATION`` / ``PUT_DOWN_DURATION`` when no CLI pairs are
    given.
    """
    out: dict[str, float] = {}
    if args_pairs:
        for p in args_pairs:
            if "=" not in p:
                raise SystemExit(
                    f"--duration expects name=seconds, got: {p!r}"
                )
            name, _, value = p.partition("=")
            name = name.strip()
            value = value.strip()
            if not name or not value:
                raise SystemExit(f"--duration expects name=seconds, got: {p!r}")
            try:
                seconds = float(value)
            except ValueError as e:
                raise SystemExit(
                    f"--duration seconds must be a number, got: {value!r}"
                ) from e
            if seconds <= 0:
                raise SystemExit(
                    f"--duration seconds must be > 0, got: {seconds}"
                )
            out[name] = seconds
        return out

    defaults = (
        ("pick_up", os.environ.get("PICK_UP_DURATION")),
        ("put_down", os.environ.get("PUT_DOWN_DURATION")),
    )
    for name, value in defaults:
        if value:
            try:
                seconds = float(value)
            except ValueError as e:
                raise SystemExit(
                    f"{name.upper()}_DURATION must be a number, got: {value!r}"
                ) from e
            if seconds > 0:
                out[name] = seconds
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
    parser.add_argument(
        "--default-policy",
        default=os.environ.get("RUN_POLICY_DEFAULT"),
        metavar="NAME",
        help="Pre-select this policy at startup so the tick loop is warm "
             "before any run_policy() RPC. Does not auto-claim the "
             "active-operator slot; a peer (or teleop `c`) still has to "
             "(env: RUN_POLICY_DEFAULT).",
    )
    parser.add_argument(
        "--reward",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Attach a sparse-reward checkpoint (e.g. best.pt from "
             "utils/train_sparse_reward) to a registered policy. "
             "Repeatable. Defaults to PICK_UP_REWARD / PUT_DOWN_REWARD "
             "env vars. Policies without a reward attached will not "
             "auto-stop on completion.",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=float(os.environ.get("REWARD_THRESHOLD", 0.7)),
        help="P(done) value above which a tick counts as 'task done' "
             "(default 0.7, env: REWARD_THRESHOLD).",
    )
    parser.add_argument(
        "--reward-trigger-ticks",
        type=int,
        default=int(os.environ.get("REWARD_TRIGGER_TICKS", 10)),
        help="How many consecutive above-threshold ticks must occur "
             "before the run_policy() RPC returns done (default 10; at "
             "30 fps that's ~0.33 s, env: REWARD_TRIGGER_TICKS).",
    )
    parser.add_argument(
        "--reward-camera",
        default=os.environ.get("REWARD_CAMERA", "arm_camera"),
        help="Camera key fed into the sparse-reward model. The shipped "
             "checkpoints train on 'arm_camera'; override only if your "
             "reward model was trained on a different stream.",
    )
    parser.add_argument(
        "--duration",
        action="append",
        default=[],
        metavar="NAME=SECONDS",
        help="Auto-stop the named policy after a fixed number of seconds, "
             "independent of any reward classifier. Useful for policies "
             "like put_down where a learned 'done' signal is overkill. "
             "If both a reward checkpoint and a duration are configured, "
             "whichever fires first parks the operator. Defaults to "
             "PICK_UP_DURATION / PUT_DOWN_DURATION env vars.",
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

    reward_paths = _parse_reward_paths(args.reward)
    unknown_reward = sorted(set(reward_paths) - set(policy_paths))
    if unknown_reward:
        raise SystemExit(
            f"--reward names not registered as policies: {unknown_reward} "
            f"(known: {sorted(policy_paths)})"
        )
    for name, path in reward_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"reward checkpoint not found: {path}")

    durations = _parse_duration_specs(args.duration)
    unknown_duration = sorted(set(durations) - set(policy_paths))
    if unknown_duration:
        raise SystemExit(
            f"--duration names not registered as policies: {unknown_duration} "
            f"(known: {sorted(policy_paths)})"
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
        reward_path = reward_paths.get(name)
        duration_s = durations.get(name)
        extras = []
        if reward_path:
            extras.append(f"+reward {reward_path}")
        if duration_s is not None:
            extras.append(f"+duration {duration_s}s")
        suffix = (" (" + ", ".join(extras) + ")") if extras else ""
        print(f"[run-policy-operator] loading '{name}' ...{suffix}")
        policies[name] = LoadedPolicy(
            name=name,
            checkpoint_dir=path,
            device=device,
            temporal_ensemble=args.temporal_ensemble,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            reward_checkpoint=reward_path,
            reward_camera=args.reward_camera,
            duration_s=duration_s,
        )
        scorer_tag = (
            f"reward={'on' if policies[name].scorer is not None else 'off'}"
        )
        duration_tag = (
            f"duration={duration_s}s" if duration_s is not None else "duration=off"
        )
        print(
            f"[run-policy-operator]   '{name}' cameras={policies[name].cameras} "
            f"chunk={policies[name].policy.config.chunk_size} "
            f"{scorer_tag} {duration_tag}"
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
    # handler / tick loop / timeout task all run on the asyncio loop),
    # so no lock needed on CPython.
    current: Optional[LoadedPolicy] = None
    # When a run_policy() RPC engages a policy that has at least one
    # terminator (reward classifier and/or fixed duration), we hand it
    # an asyncio.Future that resolves once a terminator fires. The
    # tick loop resolves it on reward; a sleep coroutine resolves it
    # on duration. Counters are reset each time `current` is (re)assigned.
    done_future: Optional[asyncio.Future] = None
    timeout_task: Optional[asyncio.Task] = None
    reward_above_streak = 0
    last_reward_prob: float = 0.0
    last_logged_policy: Optional[str] = None

    if args.default_policy:
        if args.default_policy not in policies:
            raise SystemExit(
                f"--default-policy {args.default_policy!r} not in registered "
                f"policies: {sorted(policies)}"
            )
        current = policies[args.default_policy]
        current.reset()
        print(
            f"[run-policy-operator] default policy '{args.default_policy}' "
            f"pre-selected (warm); awaiting active-operator handoff"
        )

    async def _park(result: dict) -> None:
        """Resolve the current done_future, drop the tick loop's policy,
        cancel any pending timeout, and release the active-operator
        slot. Idempotent: a second call is a no-op."""
        nonlocal current, done_future, timeout_task
        nonlocal reward_above_streak, last_logged_policy
        if done_future is None:
            return
        fut = done_future
        done_future = None
        current = None
        reward_above_streak = 0
        last_logged_policy = None
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        timeout_task = None
        if not fut.done():
            fut.set_result(result)
        try:
            await op.set_active_operator(None)
        except Exception as e:
            print(f"[run-policy-operator] release active-operator failed: {e}")

    async def _duration_watchdog(
        seconds: float, policy_name: str, target_future: asyncio.Future
    ) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        if target_future.done() or done_future is not target_future:
            # Preempted, completed by reward, or shut down before we fired.
            return
        print(
            f"[run-policy-operator] duration {seconds:.2f}s elapsed for "
            f"'{policy_name}'; parking"
        )
        await _park({
            "status": "done",
            "reason": "timeout",
            "seconds": seconds,
        })

    async def run_policy_rpc(data: RpcInvocationData) -> str:
        nonlocal current, done_future, timeout_task
        nonlocal reward_above_streak, last_reward_prob
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

        # Preempt any in-flight run_policy() so its caller gets a clean
        # reply rather than hanging on a future we're about to drop.
        if done_future is not None and not done_future.done():
            done_future.set_result({"status": "preempted", "by": name})
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        timeout_task = None

        has_terminator = chosen.scorer is not None or chosen.duration_s is not None
        new_future: Optional[asyncio.Future] = None
        if has_terminator:
            new_future = asyncio.get_running_loop().create_future()
        done_future = new_future
        reward_above_streak = 0
        last_reward_prob = 0.0
        current = chosen

        if new_future is not None and chosen.duration_s is not None:
            timeout_task = asyncio.create_task(
                _duration_watchdog(chosen.duration_s, name, new_future)
            )

        me_id = op.local_identity()
        await op.set_active_operator(me_id)
        print(
            f"[run-policy-operator] run_policy('{name}') -> "
            f"active operator -> {me_id} cameras={chosen.cameras} "
            f"reward={'on' if chosen.scorer is not None else 'off'} "
            f"duration={chosen.duration_s if chosen.duration_s is not None else 'off'}"
        )

        if new_future is None:
            # No terminators attached; preserve the pre-terminator
            # behaviour and return immediately, flagging that the
            # caller is on its own for completion detection.
            return json.dumps({
                "policy": name,
                "cameras": list(chosen.cameras),
                "active_operator": me_id,
                "status": "no_terminator",
            })

        try:
            result = await new_future
        except asyncio.CancelledError:
            # Caller-side cancellation. Park before re-raising so we
            # don't leave the tick loop driving with a dead future.
            if current is chosen:
                await _park({"status": "cancelled"})
            raise

        return json.dumps({
            "policy": name,
            "cameras": list(chosen.cameras),
            "active_operator": me_id,
            **result,
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
                # No policy selected yet; tick loop parks. The teleop
                # side or any operator can call run_policy() to engage.
                continue

            if last_logged_policy != active.name:
                last_logged_policy = active.name
                print(
                    f"[run-policy-operator] driving with '{active.name}' "
                    f"cameras={active.cameras}"
                )

            state_typed = dict(obs.state)
            frames: dict[str, np.ndarray] = {}
            needed_cams = set(active.cameras)
            if active.scorer is not None:
                needed_cams.add(active.reward_camera)
            for cam in needed_cams:
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

            if active.scorer is not None and done_future is not None:
                reward_frame = frames.get(active.reward_camera)
                if reward_frame is None:
                    # Reward camera not in this obs; drop the streak so
                    # a stale reading can't carry us across the trigger.
                    reward_above_streak = 0
                else:
                    state_vec = [float(state_typed[k]) for k in STATE_KEYS]
                    try:
                        prob = active.scorer.score(reward_frame, state_vec)
                    except Exception as e:
                        print(f"[run-policy-operator] reward error: {e}")
                        prob = 0.0
                    last_reward_prob = prob
                    if prob >= args.reward_threshold:
                        reward_above_streak += 1
                    else:
                        reward_above_streak = 0

                    if tick_count % fps == 0:
                        print(
                            f"[run-policy-operator]   reward p={prob:.3f} "
                            f"streak={reward_above_streak}/"
                            f"{args.reward_trigger_ticks}"
                        )

                    if reward_above_streak >= args.reward_trigger_ticks:
                        print(
                            f"[run-policy-operator] done: '{active.name}' "
                            f"reward streak hit "
                            f"({args.reward_trigger_ticks} ticks "
                            f"@ >= {args.reward_threshold}); parking"
                        )
                        await _park({
                            "status": "done",
                            "reason": "reward",
                            "reward_prob": prob,
                            "ticks_above": reward_above_streak,
                        })

    except KeyboardInterrupt:
        print("\n[run-policy-operator] stopping ...")
    finally:
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        if done_future is not None and not done_future.done():
            done_future.set_result({"status": "cancelled"})
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    asyncio.run(main())
