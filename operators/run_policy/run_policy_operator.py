"""Drive the remote leslider with one of N policies (ACT checkpoints or
prerecorded scripted playback).

A single Portal Operator that pre-loads multiple policies and exposes
an RPC

    run_policy(policy: "pick_up" | "put_down" | ..., side: "left" | "right")

Two policy types are supported, registered separately at startup:

  * ACT policies (``--policy NAME=PATH``, env ``<NAME>_CHECKPOINT``):
    trained via ``utils/data_collection``. Each tick runs an ACT
    forward pass over the latest camera obs and 7-d state.
  * Scripted policies (``--script NAME=PATH``, env ``<NAME>_SCRIPT``):
    play back a randomly-selected episode's recorded action sequence
    from a LeRobot v3 dataset directory. No model, no cameras read.
    Used for motions like ``put_down`` where a learned policy is
    overkill.

The RPC:

  1. Parses ``policy`` and (optional) ``side`` from the payload. The
     payload may be a bare policy-name string (side defaults to
     ``right``) or a JSON object ``{"policy": "...", "side": "..."}``.
  2. Resets the chosen policy. For scripted policies this picks a
     fresh random episode and records its length as a per-run
     duration; for ACT policies this clears the internal action
     chunk / temporal ensembler and the sparse-reward "done" counter.
  3. Claims active-operator control of the robot.
  4. Switches the continuous tick loop to drive that policy.
  5. Blocks the RPC reply until a "task done" signal fires:
       * scripted: when playback exhausts the random episode, or
         the per-run duration watchdog (episode length + 2 s safety
         buffer) fires, whichever comes first.
       * ACT: a sparse-reward classifier
         (``utils/train_sparse_reward``) reaching
         ``P(done) >= threshold`` for ``trigger_ticks`` consecutive
         ticks, and/or a fixed ``--duration`` / ``<NAME>_DURATION``
         seconds. If neither is configured, falls back to
         ``DEFAULT_DURATION`` (30 s by default, env-overridable; set
         to 0 to disable and return immediately with
         ``status: "no_terminator"``).
     On done, the tick loop parks, the active-operator slot is
     released, and the RPC returns success.

The ``side`` parameter only affects scripted policies. When
``side == "left"``, the first joint (shoulder_pan, index 0 of the
7-d action) is sign-flipped to mirror a recording captured on the
right side. ACT policies ignore ``side``; passing ``side="left"`` to
an ACT policy is a no-op with a warning log.

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
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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

# Action-vector index whose sign is flipped when a scripted policy is
# invoked with side="left". Index 0 = shoulder_pan.pos (the base
# rotation), which mirrors left-vs-right placement geometrically.
SCRIPTED_FLIP_INDEX = 0
# Safety-net buffer added on top of (episode_frames / fps) when arming
# the per-run duration watchdog for a scripted playback. Natural script
# end normally fires first; the watchdog only catches pathological
# tick-loop stalls.
SCRIPTED_DURATION_BUFFER_S = 2.0

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

        self.is_scripted = False

    def reset(self, side: str = "right") -> None:
        del side
        self.policy.reset()


def _load_episode_actions(dataset_dir: Path) -> dict[int, np.ndarray]:
    """Load per-episode action sequences from a LeRobot v3 dataset dir.

    Reads every ``data/chunk-*/file-*.parquet`` under ``dataset_dir`` and
    groups rows by ``episode_index``. Returns
    ``{episode_index: (T, 7) float32 ndarray}``.

    Action rows are stored as parquet ``list<float>``; we flatten the
    whole column in one shot and ``np.split`` on episode boundaries
    rather than converting each row through Python, which is ~15x
    faster on the put_down dataset (13k rows, ~70 ms -> ~5 ms)."""
    data_root = dataset_dir / "data"
    parquet_paths = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"no parquet files under {data_root}; expected a LeRobot v3 dataset"
        )
    tables = [pq.read_table(p, columns=["episode_index", "action"]) for p in parquet_paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    action_col = table["action"].combine_chunks()
    n_rows = len(action_col)
    if n_rows == 0:
        return {}
    flat = action_col.values.to_numpy(zero_copy_only=False).astype(
        np.float32, copy=False
    )
    width = flat.size // n_rows
    actions = flat.reshape(n_rows, width)

    ep_idx = table["episode_index"].to_numpy().astype(np.int64, copy=False)
    # Rows in LeRobot v3 datasets are written in (episode_index, frame_index)
    # order, so episode runs are contiguous. Split where episode_index
    # changes; first index of each run gives the key.
    change = np.flatnonzero(np.diff(ep_idx)) + 1
    splits = np.split(actions, change)
    ep_keys = ep_idx[np.r_[0, change]]
    return {int(e): a for e, a in zip(ep_keys, splits)}


class ScriptedPolicy:
    """Prerecorded-motion "policy" that plays back one randomly-selected
    episode from a LeRobot v3 dataset per ``reset()``. Exposes the same
    duck-typed surface (``name``, ``cameras``, ``scorer``, ``reward_camera``,
    ``duration_s``, ``reset()``) as :class:`LoadedPolicy` so the tick loop
    and RPC handler can hold either kind in the same dict.

    No model, no inference, no cameras read. The tick loop fast-paths
    scripted policies via ``is_scripted=True`` and ``next_action()``.

    ``side="left"`` at reset flips the sign of action index
    ``SCRIPTED_FLIP_INDEX`` (shoulder_pan) on every emitted action so a
    recording captured on the right side mirrors to the left."""

    def __init__(self, name: str, dataset_dir: Path, fps: int) -> None:
        self.name = name
        self.dataset_dir = dataset_dir
        self.fps = fps
        self.episodes = _load_episode_actions(dataset_dir)
        if not self.episodes:
            raise ValueError(f"scripted dataset {dataset_dir} has no episodes")
        self.episode_keys = sorted(self.episodes.keys())
        self.rng = random.Random()

        # Duck-typed surface for the existing tick loop / RPC code.
        self.cameras: tuple[str, ...] = ()
        self.scorer = None
        self.reward_camera = ""
        self.is_scripted = True

        # Per-run state, (re)assigned by every reset().
        self.actions: Optional[np.ndarray] = None
        self.cursor: int = 0
        self.flip_pan: bool = False
        self.episode_index: int = -1
        # duration_s is rearmed per-run from the chosen episode length;
        # the RPC handler reads it AFTER reset() to set the watchdog.
        self.duration_s: Optional[float] = None

    def reset(self, side: str = "right") -> None:
        self.episode_index = self.rng.choice(self.episode_keys)
        self.actions = self.episodes[self.episode_index]
        self.cursor = 0
        self.flip_pan = side == "left"
        self.duration_s = len(self.actions) / float(self.fps) + SCRIPTED_DURATION_BUFFER_S

    def next_action(self) -> Optional[dict[str, float]]:
        """Advance the cursor by one and return the next action dict, or
        ``None`` when the recorded episode is exhausted (signals
        ``script_end`` to the tick loop)."""
        if self.actions is None or self.cursor >= len(self.actions):
            return None
        a = self.actions[self.cursor]
        self.cursor += 1
        if self.flip_pan:
            a = a.copy()
            a[SCRIPTED_FLIP_INDEX] = -a[SCRIPTED_FLIP_INDEX]
        return {k: float(a[i]) for i, k in enumerate(ACTION_KEYS)}


AnyPolicy = Union[LoadedPolicy, "ScriptedPolicy"]


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


def _parse_script_paths(args_pairs: list[str]) -> dict[str, Path]:
    """Parse ``name=path`` pairs into a mapping of scripted-policy dataset
    directories. Falls back to env var ``PUT_DOWN_SCRIPT`` when no CLI
    pairs are given. Each path must be a LeRobot v3 dataset directory
    (i.e. contains ``data/chunk-*/file-*.parquet``)."""
    out: dict[str, Path] = {}
    if args_pairs:
        for p in args_pairs:
            if "=" not in p:
                raise SystemExit(f"--script expects name=path, got: {p!r}")
            name, _, path = p.partition("=")
            name = name.strip()
            if not name or not path.strip():
                raise SystemExit(f"--script expects name=path, got: {p!r}")
            out[name] = Path(path).expanduser().resolve()
        return out

    defaults = (("put_down", os.environ.get("PUT_DOWN_SCRIPT")),)
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
        help="Register a named ACT policy checkpoint. Repeatable. "
             "Defaults to PICK_UP_CHECKPOINT / PUT_DOWN_CHECKPOINT env vars.",
    )
    parser.add_argument(
        "--script",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register a named SCRIPTED policy backed by a prerecorded "
             "LeRobot v3 dataset directory. The chosen episode is "
             "randomized per run_policy() call and its length sets that "
             "run's RPC timeout. Repeatable. Defaults to PUT_DOWN_SCRIPT "
             "env var. If a name is given to both --policy and --script, "
             "--script wins.",
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
             "PICK_UP_DURATION / PUT_DOWN_DURATION env vars. Any policy "
             "without an explicit duration or reward classifier falls "
             "back to DEFAULT_DURATION (30s; set to 0 to disable).",
    )
    parser.set_defaults(temporal_ensemble=True)
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    policy_paths = _parse_policy_paths(args.policy)
    script_paths = _parse_script_paths(args.script)

    # Scripted always wins over ACT for the same name (env or CLI). The
    # tick loop's behaviour for a name diverges by type, so they cannot
    # coexist under one name.
    for name in list(policy_paths.keys()):
        if name in script_paths:
            print(
                f"[run-policy-operator] note: '{name}' registered as both "
                f"ACT and scripted; using scripted ({script_paths[name]})"
            )
            del policy_paths[name]

    if not policy_paths and not script_paths:
        raise SystemExit(
            "no policies registered; pass --policy name=path or "
            "--script name=path (or set PICK_UP_CHECKPOINT / "
            "PUT_DOWN_CHECKPOINT / PUT_DOWN_SCRIPT)"
        )

    all_policy_names = set(policy_paths) | set(script_paths)

    reward_paths = _parse_reward_paths(args.reward)
    unknown_reward = sorted(set(reward_paths) - all_policy_names)
    if unknown_reward:
        raise SystemExit(
            f"--reward names not registered as policies: {unknown_reward} "
            f"(known: {sorted(all_policy_names)})"
        )
    # Scripted policies own their own terminator (script end + per-run
    # duration watchdog); a reward classifier would be redundant and
    # confusing. Silently drop env-default rewards that target scripted
    # names so PUT_DOWN_REWARD left over from an earlier ACT setup
    # doesn't bite.
    for name in list(reward_paths.keys()):
        if name in script_paths:
            print(
                f"[run-policy-operator] note: reward checkpoint for "
                f"scripted policy '{name}' ignored"
            )
            del reward_paths[name]
    for name, path in reward_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"reward checkpoint not found: {path}")

    durations = _parse_duration_specs(args.duration)
    unknown_duration = sorted(set(durations) - all_policy_names)
    if unknown_duration:
        raise SystemExit(
            f"--duration names not registered as policies: {unknown_duration} "
            f"(known: {sorted(all_policy_names)})"
        )
    for name in list(durations.keys()):
        if name in script_paths:
            print(
                f"[run-policy-operator] note: --duration for scripted "
                f"policy '{name}' ignored (uses random episode length)"
            )
            del durations[name]

    try:
        default_duration_s: Optional[float] = float(
            os.environ.get("DEFAULT_DURATION", "30")
        )
    except ValueError as e:
        raise SystemExit(
            f"DEFAULT_DURATION must be a number, got: "
            f"{os.environ.get('DEFAULT_DURATION')!r}"
        ) from e
    if default_duration_s is not None and default_duration_s <= 0:
        default_duration_s = None

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
        print(f"[run-policy-operator]   policy '{name}' (ACT) <- {path}")
    for name, path in script_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"scripted dataset directory not found: {path}")
        print(f"[run-policy-operator]   policy '{name}' (scripted) <- {path}")

    policies: dict[str, AnyPolicy] = {}
    for name, path in policy_paths.items():
        reward_path = reward_paths.get(name)
        if reward_path is not None:
            duration_s = None
        else:
            duration_s = durations.get(name, default_duration_s)
        extras = []
        if reward_path:
            extras.append(f"+reward {reward_path}")
        if duration_s is not None:
            extras.append(f"+duration {duration_s}s")
        suffix = (" (" + ", ".join(extras) + ")") if extras else ""
        print(f"[run-policy-operator] loading '{name}' (ACT) ...{suffix}")
        loaded = LoadedPolicy(
            name=name,
            checkpoint_dir=path,
            device=device,
            temporal_ensemble=args.temporal_ensemble,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            reward_checkpoint=reward_path,
            reward_camera=args.reward_camera,
            duration_s=duration_s,
        )
        policies[name] = loaded
        scorer_tag = f"reward={'on' if loaded.scorer is not None else 'off'}"
        duration_tag = (
            f"duration={duration_s}s" if duration_s is not None else "duration=off"
        )
        print(
            f"[run-policy-operator]   '{name}' cameras={loaded.cameras} "
            f"chunk={loaded.policy.config.chunk_size} "
            f"{scorer_tag} {duration_tag}"
        )

    for name, path in script_paths.items():
        print(f"[run-policy-operator] loading '{name}' (scripted) ...")
        scripted = ScriptedPolicy(name=name, dataset_dir=path, fps=fps)
        policies[name] = scripted
        ep_lens = {ep: len(scripted.episodes[ep]) for ep in scripted.episode_keys}
        min_f = min(ep_lens.values())
        max_f = max(ep_lens.values())
        print(
            f"[run-policy-operator]   '{name}' episodes={len(scripted.episode_keys)} "
            f"frames={min_f}-{max_f} "
            f"duration={min_f / fps:.1f}-{max_f / fps:.1f}s "
            f"(+{SCRIPTED_DURATION_BUFFER_S:.0f}s watchdog buffer)"
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
    current: Optional[AnyPolicy] = None
    # Serializes active-operator transitions between _park's release
    # (set_active_operator(None)) and a fresh run_policy()'s claim
    # (set_active_operator(me_id)). Without this, if a watchdog or
    # script_end fires at the same moment a caller re-invokes
    # run_policy, the two Portal calls race and the slot can end up
    # cleared right after we claimed it. The check inside _park's lock
    # acquire (`if current is None`) lets us skip the release entirely
    # when a new RPC has already taken over.
    op_lock = asyncio.Lock()
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
        async with op_lock:
            if current is not None:
                # A new run_policy() RPC raced ahead and claimed the
                # slot while we were waiting on the lock. Let it own
                # the active-operator state; releasing now would
                # immediately unclaim what was just claimed.
                return
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
        caller = getattr(data, "caller_identity", None)
        print(
            f"[run-policy-operator] RPC run_policy received "
            f"from={caller!r} payload={data.payload!r}"
        )
        payload = (data.payload or "").strip()
        name = payload
        side_raw: Optional[str] = None
        if payload.startswith("{"):
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                raise RpcError.Error(
                    code=1400,
                    message="payload was not valid JSON; pass a bare policy "
                            "name string instead",
                    data=None,
                )
            # `or ""` so `{"policy": null}` collapses to "" and hits
            # the empty-name branch instead of `str(None) == "None"`
            # and a confusing 1404.
            name = str(obj.get("policy") or "").strip()
            raw = obj.get("side")
            if raw is not None:
                side_raw = str(raw).strip().lower()
        side = side_raw if side_raw else "right"
        if side not in ("left", "right"):
            raise RpcError.Error(
                code=1400,
                message=f"invalid side {side!r}; expected 'left' or 'right'",
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
        if side == "left" and not getattr(chosen, "is_scripted", False):
            print(
                f"[run-policy-operator] warning: side='left' has no effect on "
                f"non-scripted policy '{name}'"
            )

        # Preempt any in-flight run_policy() BEFORE mutating chosen's
        # state, so the old run's caller gets a clean "preempted" reply
        # and its watchdog is cancelled before we reset the policy.
        # Doing this in the other order works today only because no
        # await sits between reset() and the preempt block; any future
        # refactor that adds one would let the tick loop observe a
        # half-reset policy.
        if done_future is not None and not done_future.done():
            done_future.set_result({"status": "preempted", "by": name})
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        timeout_task = None

        # reset() must run AFTER preempt (see above) but BEFORE the
        # has_terminator check, since ScriptedPolicy.reset is what
        # arms chosen.duration_s for this run.
        chosen.reset(side=side)

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
        async with op_lock:
            await op.set_active_operator(me_id)
        scripted_tag = ""
        if getattr(chosen, "is_scripted", False):
            scripted_tag = (
                f" episode={chosen.episode_index} "
                f"frames={len(chosen.actions)} "
                f"side={side}"
            )
        print(
            f"[run-policy-operator] run_policy('{name}', side={side}) -> "
            f"active operator -> {me_id} cameras={chosen.cameras} "
            f"reward={'on' if chosen.scorer is not None else 'off'} "
            f"duration={chosen.duration_s if chosen.duration_s is not None else 'off'}"
            f"{scripted_tag}"
        )

        base_response: dict = {
            "policy": name,
            "side": side,
            "cameras": list(chosen.cameras),
            "active_operator": me_id,
        }
        if getattr(chosen, "is_scripted", False):
            base_response["episode_index"] = int(chosen.episode_index)
            base_response["frames"] = int(len(chosen.actions))

        if new_future is None:
            # No terminators attached; preserve the pre-terminator
            # behaviour and return immediately, flagging that the
            # caller is on its own for completion detection.
            response = json.dumps({**base_response, "status": "no_terminator"})
            print(
                f"[run-policy-operator] RPC run_policy('{name}') -> "
                f"returning to {caller!r}: {response}"
            )
            return response

        try:
            result = await new_future
        except asyncio.CancelledError:
            # Caller-side cancellation. Park before re-raising so we
            # don't leave the tick loop driving with a dead future.
            if current is chosen:
                await _park({"status": "cancelled"})
            raise

        response = json.dumps({**base_response, **result})
        print(
            f"[run-policy-operator] RPC run_policy('{name}') -> "
            f"returning to {caller!r}: {response}"
        )
        return response

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
            else:
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

            if active.is_scripted:
                # Scripted playback doesn't read obs.state or obs.frames,
                # so we don't gate on obs being present. If the robot is
                # alive but its camera is hung, scripted runs still
                # advance. in_reply_to_ts_us falls back to 0 (= "no
                # preceding observation") when there's no obs yet.
                action_dict = active.next_action()
                if action_dict is None:
                    print(
                        f"[run-policy-operator] done: '{active.name}' "
                        f"script_end episode={active.episode_index} "
                        f"frames={active.cursor}; parking"
                    )
                    await _park({
                        "status": "done",
                        "reason": "script_end",
                        "episode_index": int(active.episode_index),
                        "frames": int(active.cursor),
                    })
                    continue
                op.send_action(
                    action_dict,
                    timestamp_us=int(time.time() * 1_000_000),
                    in_reply_to_ts_us=obs.timestamp_us if obs is not None else 0,
                )
                tick_count += 1
                continue

            # ACT path needs obs (state + camera frames).
            if obs is None:
                continue

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
