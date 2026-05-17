"""Drive the remote leslider autonomously using a trained Diffusion Policy checkpoint.

Mirrors policies/act/inference.py but for Diffusion Policy. Diffusion
replans every ``n_action_steps`` ticks (e.g. 8 = ~0.27s at 30Hz) and
runs N denoising steps per replan.

Knobs:
  --num-inference-steps N   Override denoising steps at inference (default: trained
                            value, typically 100 for DDPM). Lower = faster but worse.
                            DDIM-trained checkpoints can go down to ~10.
  --scheduler {DDPM,DDIM}   Override scheduler at inference (default: keep trained type).
  --async-predict           Run the next chunk's forward pass on a background thread.
  --blend                   Cross-fade between consecutive chunks. Requires --async-predict.
  --anchor-prefix N         Inpaint first N positions of each new chunk to match the
                            previous chunk's predicted next-N actions. Requires --async.

Hotkeys (starts DISENGAGED):
  SPACE: toggle autonomous control on/off.
  x:     clean quit.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import select
import sys
import termios
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
from lerobot.policies.utils import populate_queues
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import ACTION as ACTION_KEY, OBS_IMAGES, OBS_STATE
from lerobot.utils.visualization_utils import init_rerun

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    frame_bytes_to_numpy_rgb,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from _common import env_int, env_str, load_env, mint_token, pace, required_env  # noqa: E402

IDENTITY = "policy-diffusion"
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "portal.yaml"

# 7-dim state and action (alphabetical, set at dataset creation time).
STATE_KEYS = (
    "elbow_flex.pos",
    "gripper.pos",
    "shoulder_lift.pos",
    "shoulder_pan.pos",
    "slider.vel",
    "wrist_flex.pos",
    "wrist_roll.pos",
)
ACTION_KEYS = STATE_KEYS

logger = logging.getLogger(__name__)


class AsyncChunkPolicy:
    """Wraps a DiffusionPolicy with two improvements over stock select_action:

      1. **Async chunk prefetch**, the next chunk's forward pass runs on a
         background thread, overlapping with the current chunk's dispatch.
         Eliminates the freeze every ``n_action_steps`` ticks.

      2. **Overlap blending**, each prediction returns the full ``horizon``
         actions, not just ``n_action_steps``. The "extra" ``n_action_steps``
         tail of the previous chunk overlaps in time with the head of the
         new chunk; we cross-fade between them so chunk boundaries don't
         produce action discontinuities.

    Anchor prefix (optional) inpaints the start of each new chunk to
    match the previous chunk's committed next-N actions, eliminating
    mode-flip jumps when the model is multi-modal.
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        kick_off_at_remaining: int = 2,
        blend: bool = False,
        anchor_len: int = 0,
    ):
        self.policy = policy
        self.config = policy.config
        self.n_action_steps = policy.config.n_action_steps
        self.horizon = policy.config.horizon
        self._action_start = policy.config.n_obs_steps - 1
        self._extended_len = self.horizon - self._action_start
        self.kick_off_at_remaining = kick_off_at_remaining
        self.blend = blend
        self.anchor_len = max(0, min(anchor_len, self._extended_len - self.n_action_steps))
        self._extended_chunk = self.blend or self.anchor_len > 0

        param = next(policy.diffusion.parameters())
        self._device = param.device
        self._dtype = param.dtype
        self._action_dim = policy.config.action_feature.shape[0]

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._pending_anchor = None
        self._current_chunk = None
        self._prev_chunk = None
        self._idx = 0

    def reset(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
        self._current_chunk = None
        self._prev_chunk = None
        self._idx = 0
        self.policy.reset()

    def to(self, *a, **kw):
        return self.policy.to(*a, **kw)

    def eval(self):
        return self.policy.eval()

    def _snapshot_obs_batch(self) -> dict:
        return {
            k: torch.stack(list(q), dim=1)
            for k, q in self.policy._queues.items()
            if k != ACTION_KEY
        }

    def _run_predict(
        self, snapshot_batch: dict, anchor: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.inference_mode():
            diffusion = self.policy.diffusion
            scheduler = diffusion.noise_scheduler
            batch_size = snapshot_batch[OBS_STATE].shape[0]
            global_cond = diffusion._prepare_global_conditioning(snapshot_batch)

            sample = torch.randn(
                size=(batch_size, self.horizon, self._action_dim),
                dtype=self._dtype,
                device=self._device,
            )

            scheduler.set_timesteps(diffusion.num_inference_steps)

            anchor_batched = None
            anchor_n = 0
            anchor_at = self._action_start
            if anchor is not None and anchor.numel() > 0:
                anchor_batched = anchor.unsqueeze(0).to(self._device, self._dtype)
                anchor_n = anchor_batched.shape[1]

            for t in scheduler.timesteps:
                if anchor_n > 0:
                    t_arr = torch.tensor([t], device=self._device, dtype=torch.long)
                    eps = torch.randn_like(anchor_batched)
                    noisy_anchor = scheduler.add_noise(anchor_batched, eps, t_arr)
                    sample[:, anchor_at : anchor_at + anchor_n] = noisy_anchor

                model_output = diffusion.unet(
                    sample,
                    torch.full(sample.shape[:1], t, dtype=torch.long, device=self._device),
                    global_cond=global_cond,
                )
                sample = scheduler.step(model_output, t, sample).prev_sample

            if anchor_n > 0:
                sample[:, anchor_at : anchor_at + anchor_n] = anchor_batched

            actions = sample[:, self._action_start :]
            if not self._extended_chunk:
                actions = actions[:, : self.n_action_steps]
            return actions[0]

    def _build_anchor(self) -> torch.Tensor | None:
        if self.anchor_len == 0 or self._current_chunk is None:
            return None
        start = self.n_action_steps
        end = start + self.anchor_len
        if end > self._current_chunk.shape[0]:
            return None
        return self._current_chunk[start:end]

    def _maybe_kickoff_async(self) -> None:
        ticks_left = self.n_action_steps - self._idx
        if (
            self._pending_future is None
            and 0 < ticks_left <= self.kick_off_at_remaining
        ):
            self._pending_anchor = self._build_anchor()
            self._pending_future = self._executor.submit(
                self._run_predict, self._snapshot_obs_batch(), self._pending_anchor
            )

    def _swap_to_next_chunk(self) -> None:
        if self._pending_future is not None:
            next_chunk = self._pending_future.result()
            self._pending_future = None
            self._pending_anchor = None
        else:
            anchor = self._build_anchor()
            next_chunk = self._run_predict(self._snapshot_obs_batch(), anchor)
        self._prev_chunk = self._current_chunk
        self._current_chunk = next_chunk
        self._idx = 0

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        batch = dict(batch)
        if ACTION_KEY in batch:
            batch.pop(ACTION_KEY)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        self.policy._queues = populate_queues(self.policy._queues, batch)

        if self._current_chunk is None:
            self._current_chunk = self._run_predict(self._snapshot_obs_batch())
            self._idx = 0

        if self._idx >= self.n_action_steps:
            self._swap_to_next_chunk()

        new_action = self._current_chunk[self._idx]

        if self.blend:
            prev_idx = self._idx + self.n_action_steps
            if (
                self._prev_chunk is not None
                and prev_idx < self._prev_chunk.shape[0]
            ):
                old_action = self._prev_chunk[prev_idx]
                alpha = self._idx / max(self.n_action_steps - 1, 1)
                action = alpha * new_action + (1.0 - alpha) * old_action
            else:
                action = new_action
        else:
            action = new_action

        self._idx += 1
        self._maybe_kickoff_async()
        return action.unsqueeze(0)


@contextmanager
def _raw_stdin():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _policy_cameras(policy) -> tuple[str, ...]:
    keys = getattr(policy.config, "input_features", {}) or {}
    cams = tuple(
        k.removeprefix("observation.images.")
        for k in keys
        if k.startswith("observation.images.")
    )
    return cams or ("arm_camera",)


def _build_batch(state_typed, frames, policy_cameras, task_prompt) -> dict:
    state = torch.tensor([float(state_typed[k]) for k in STATE_KEYS], dtype=torch.float32)
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


def _rr_log(entity, data, **kw) -> None:
    try:
        rr.log(entity, data, **kw)
    except Exception:
        pass


def _build_blueprint(cameras: tuple[str, ...]) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*(rrb.Spatial2DView(origin=f"/observation/{c}", name=c) for c in cameras)),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/observation", name="State"),
                rrb.TimeSeriesView(origin="/action", name="Predicted action"),
                rrb.TimeSeriesView(origin="/policy", name="Engagement"),
            ),
            row_shares=[2, 1],
        ),
        collapse_panels=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diffusion Policy inference over LiveKit Portal.")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DIFFUSION_CHECKPOINT", "checkpoints/diffusion-leslider/025000"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--scheduler", choices=("DDPM", "DDIM"), default=None)
    parser.add_argument("--async-predict", action="store_true")
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--anchor-prefix", type=int, default=0)
    parser.add_argument("--no-claim", dest="self_claim", action="store_false")
    parser.set_defaults(self_claim=True)
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

    print(f"[policy-diffusion] device={device} checkpoint={checkpoint_dir}")

    policy = DiffusionPolicy.from_pretrained(str(checkpoint_dir))
    policy.to(device)
    policy.eval()
    policy_cameras = _policy_cameras(policy)
    print(f"[policy-diffusion] cameras (from checkpoint): {policy_cameras}")

    if args.scheduler is not None and args.scheduler != policy.config.noise_scheduler_type:
        cfg_p = policy.config
        new_sched = _make_noise_scheduler(
            args.scheduler,
            num_train_timesteps=cfg_p.num_train_timesteps,
            beta_start=cfg_p.beta_start, beta_end=cfg_p.beta_end,
            beta_schedule=cfg_p.beta_schedule,
            clip_sample=cfg_p.clip_sample, clip_sample_range=cfg_p.clip_sample_range,
            prediction_type=cfg_p.prediction_type,
        )
        policy.diffusion.noise_scheduler = new_sched
        cfg_p.noise_scheduler_type = args.scheduler
        print(f"[policy-diffusion] override scheduler -> {args.scheduler}")

    if args.num_inference_steps is not None:
        if hasattr(policy, "diffusion"):
            policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
        print(f"[policy-diffusion] override num_inference_steps -> {args.num_inference_steps}")
    print(
        f"[policy-diffusion] horizon={policy.config.horizon} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"n_obs_steps={policy.config.n_obs_steps} "
        f"scheduler={policy.config.noise_scheduler_type}"
    )

    if args.async_predict:
        policy = AsyncChunkPolicy(policy, blend=args.blend, anchor_len=args.anchor_prefix)
        print(f"[policy-diffusion] async ON, blend={args.blend}, anchor_prefix={policy.anchor_len}")
    elif args.anchor_prefix > 0 or args.blend:
        print("[policy-diffusion] --anchor-prefix / --blend require --async-predict, ignored")

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
    op.on_active_operator_changed(lambda i: print(f"[policy-diffusion] active operator now: {i}"))

    print(f"[policy-diffusion] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()
    if args.self_claim:
        await op.set_active_operator(me)
        print(f"[policy-diffusion] self-claimed control as '{me}'")

    init_rerun(session_name=f"policy-diffusion-{room}")
    rr.send_blueprint(_build_blueprint(policy_cameras))

    print(f"[policy-diffusion] '{IDENTITY}' in '{room}' @ {fps} fps")
    print("[policy-diffusion] SPACE=engage/disengage   x=quit   (starts DISENGAGED)")

    engaged = False
    obs_keys_logged = False
    empty_obs_count = 0
    fwd_pass_count = 0
    fwd_pass_ms_sum = 0.0
    FWD_PASS_THRESHOLD_MS = 5.0
    policy.reset()
    tick_count = 0

    try:
      with _raw_stdin():
        async for _ in pace(fps):
            key = _poll_key()
            if key in ("x", "\x03"):
                break
            if key == " ":
                engaged = not engaged
                if engaged:
                    policy.reset()
                print(f"[policy-diffusion] {'ENGAGED' if engaged else 'disengaged'}")

            obs = latest_obs
            if obs is None:
                empty_obs_count += 1
                if empty_obs_count in (30, 150, 600):
                    print(
                        f"[policy-diffusion] no observations for {empty_obs_count // 30}s, "
                        "is robot.py running?"
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
                    f"[policy-diffusion] first obs: state_keys={sorted(state_typed)} "
                    f"frame_keys={sorted(frames)}"
                )

            try:
                rr.set_time("robot_time", timestamp=obs.timestamp_us / 1e6)
                for cam, frame in frames.items():
                    _rr_log(f"observation/{cam}", rr.Image(frame).compress(), static=True)
                for k, v in state_typed.items():
                    _rr_log(f"observation/{k}", rr.Scalars(float(v)))
                _rr_log("policy/engaged", rr.Scalars(float(engaged)))
            except Exception:
                pass

            if not engaged:
                continue

            batch = _build_batch(state_typed, frames, policy_cameras, task_prompt)
            try:
                processed = preprocessor(batch)
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action_t = policy.select_action(processed)
                inf_ms = (time.perf_counter() - t_inf) * 1000
                action_t = postprocessor(action_t)
            except Exception as e:
                print(f"[policy-diffusion] inference error: {e}")
                continue

            tick_count += 1
            if inf_ms > FWD_PASS_THRESHOLD_MS:
                fwd_pass_count += 1
                if fwd_pass_count > 1:
                    fwd_pass_ms_sum += inf_ms

            if tick_count % fps == 0:
                budget = (1000 / fps) * policy.config.n_action_steps
                steady_n = max(fwd_pass_count - 1, 0)
                if steady_n > 0:
                    avg_fwd = fwd_pass_ms_sum / steady_n
                    marker = "OK" if avg_fwd < budget else "OVER"
                    print(
                        f"[policy-diffusion] tick {tick_count:>5}: forward_pass "
                        f"avg={avg_fwd:6.1f}ms (n={steady_n}) "
                        f"chunk_budget={budget:.1f}ms [{marker}]"
                    )

            action = _action_dict_from_tensor(action_t)
            op.send_action(
                action,
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us,
            )
            for k, v in action.items():
                _rr_log(f"action/{k}", rr.Scalars(float(v)))

    except KeyboardInterrupt:
        print("\n[policy-diffusion] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


if __name__ == "__main__":
    asyncio.run(main())
