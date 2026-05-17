"""Drive the remote leslider autonomously using a trained ACT checkpoint.

Connects to the leslider room as a Portal `Operator`, subscribes to the
synchronized observation stream, runs `ACTPolicy.select_action(batch)`
on each frame, and sends the resulting action back as one Portal action
per tick. ACT internally manages its own action chunk + temporal
ensembler, `select_action` returns a single per-tick action no matter
how the model was trained.

Control is gated server-side: the teleop window picks the active
operator (`c` to cycle). We forward an action every tick unconditionally
— when the human is active, our `send_action` calls are dropped at the
gate. There is no local engage/disengage.

State / action conventions (must match the dataset used to train —
order is the SO101 follower's native order from `info.json`, NOT
alphabetical):

  observation.state: 7-dim:
    [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
     wrist_flex.pos, wrist_roll.pos, gripper.pos, slider.vel]

  action: 7-dim (same order):
    [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
     wrist_flex.pos, wrist_roll.pos, gripper.pos, slider.vel]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
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
    frame_bytes_to_numpy_rgb,
)

# Sibling-import: this file lives in `policies/act/`, _common.py and
# portal.yaml are at the project root. Add the root to sys.path before
# the project import.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from _common import env_int, env_str, load_env, mint_token, pace, required_env  # noqa: E402

IDENTITY = "policy-act"
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "portal.yaml"

# 7-dim state and action, in the dataset's native order (from
# data/local/portal-hitl/meta/info.json `features.observation.state.names`
# and `features.action.names`). Must match the order used when stats
# were computed — the normalizer's mean/std vectors are positional.
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


def _policy_cameras(policy) -> tuple[str, ...]:
    """Cameras the loaded checkpoint expects, in declaration order, pulled
    from `input_features`. Falls back to the single `arm_camera` if the
    checkpoint declares none."""
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
    """Build a lerobot-style batch from a state dict + camera frames."""
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
            t = t.permute(2, 0, 1)  # HWC → CHW
        batch[f"observation.images.{cam}"] = t.contiguous()
    batch["task"] = task_prompt
    return batch


def _action_dict_from_tensor(action_tensor: torch.Tensor) -> dict[str, float]:
    a = action_tensor.detach().to("cpu").squeeze().tolist()
    return {k: float(a[i]) for i, k in enumerate(ACTION_KEYS)}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACT inference over LiveKit Portal.")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "ACT_CHECKPOINT",
            "checkpoints/act-leslider/025000",
        ),
        help="Path to the pretrained_model checkpoint dir (env: ACT_CHECKPOINT)",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        dest="temporal_ensemble",
        action="store_false",
        help="Disable temporal ensembling. With TE off the policy executes a "
             "full chunk before re-planning; with TE on it re-plans every "
             "tick and weighted-averages overlapping predictions.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="Decay coefficient for temporal ensemble (default 0.01). Lower = "
             "smoother but laggier.",
    )
    parser.set_defaults(temporal_ensemble=True)
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)
    fps = env_int("PORTAL_FPS", 30)
    device = _select_device()
    task_prompt = env_str("LESLIDER_TASK", "Pick the blue cube up and hold it")

    print(f"[policy-act] device={device} checkpoint={checkpoint_dir}")

    policy = ACTPolicy.from_pretrained(str(checkpoint_dir))
    policy.to(device)
    policy.eval()
    policy_cameras = _policy_cameras(policy)
    print(f"[policy-act] cameras (from checkpoint): {policy_cameras}")

    if args.temporal_ensemble:
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble_coeff, policy.config.chunk_size
        )
        print(
            f"[policy-act] temporal ensemble ON (coeff={args.temporal_ensemble_coeff}, "
            f"chunk={policy.config.chunk_size}), forward pass every tick"
        )
    else:
        policy.config.temporal_ensemble_coeff = None
        print(
            f"[policy-act] temporal ensemble OFF, chunked execution "
            f"(forward pass every {policy.config.n_action_steps} ticks)"
        )

    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    latest_obs: Optional[Observation] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs

    op.on_observation(on_observation)
    op.on_active_operator_changed(
        lambda i: print(f"[policy-act] active operator now: {i}")
    )

    print(f"[policy-act] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()

    print(f"[policy-act] '{IDENTITY}' in '{room}' @ {fps} fps as '{me}' "
          "(teleop drives handoff via `c`)")

    obs_keys_logged = False
    empty_obs_count = 0
    fwd_pass_count = 0
    fwd_pass_ms_sum = 0.0
    FWD_PASS_THRESHOLD_MS = 5.0
    policy.reset()
    tick_count = 0

    try:
        async for _ in pace(fps):
            obs = latest_obs
            if obs is None:
                empty_obs_count += 1
                if empty_obs_count in (30, 150, 600):
                    print(
                        f"[policy-act] no observations for {empty_obs_count // 30}s, "
                        "is robot.py running and connected to the same room?"
                    )
                continue
            empty_obs_count = 0

            state_typed = dict(obs.state)
            frames = {}
            for cam in policy_cameras:
                vf = obs.frames.get(cam)
                if vf is not None:
                    frames[cam] = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)

            if not obs_keys_logged:
                obs_keys_logged = True
                print(
                    f"[policy-act] first obs: state_keys={sorted(state_typed)} "
                    f"frame_keys={sorted(frames)}"
                )

            batch = _build_batch(state_typed, frames, policy_cameras, task_prompt)
            try:
                processed = preprocessor(batch)
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action_t = policy.select_action(processed)
                t_post = time.perf_counter()
                action_t = postprocessor(action_t)
            except Exception as e:
                print(f"[policy-act] inference error: {e}")
                continue

            inf_ms = (t_post - t_inf) * 1000
            tick_count += 1
            if inf_ms > FWD_PASS_THRESHOLD_MS:
                fwd_pass_count += 1
                if fwd_pass_count > 1:  # skip the cold-start compile pass
                    fwd_pass_ms_sum += inf_ms

            if tick_count % fps == 0:  # once per second
                budget = 1000 / fps
                steady_n = max(fwd_pass_count - 1, 0)
                if steady_n > 0:
                    avg_fwd = fwd_pass_ms_sum / steady_n
                    marker = "OK" if avg_fwd < budget else "OVER"
                    print(
                        f"[policy-act] tick {tick_count:>5}: forward_pass "
                        f"avg={avg_fwd:5.1f}ms (n={steady_n}) "
                        f"budget={budget:.1f}ms [{marker}]"
                    )

            action = _action_dict_from_tensor(action_t)
            print(f"[policy-act] action: {action}")
            # `in_reply_to_ts_us` lets the robot side report
            # `metrics.policy.e2e_us_*` (true obs→action latency, not
            # network ping). When the human is the active operator the
            # gate drops this call.
            op.send_action(
                action,
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us,
            )

    except KeyboardInterrupt:
        print("\n[policy-act] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    asyncio.run(main())
