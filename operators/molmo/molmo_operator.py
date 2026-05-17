"""Drive the remote leslider autonomously using MolmoAct2-SO100_101.

MolmoAct2 is Ai2's foundation VLA. The fine-tuned checkpoint
``allenai/MolmoAct2-SO100_101`` expects the 6-DOF SO-101 follower with
two RGB cameras. Our rig is the same arm bolted onto a linear slider,
so state and action are 7-dim (the extra field is ``slider.vel``). The
model can't predict a slider command, and the slider isn't in the
SO-100/101 training distribution, so we:

  * strip ``slider.vel`` from the observation before building the model
    input,
  * append ``slider.vel = 0.0`` to every action so the slider holds
    still while the arm runs autonomously. (Slider control stays with
    the human teleop side via the leader's arrow keys; during policy
    handoff the slider just parks.)

State conversion (rig → policy), keeping the SO-101 native order. This
matches the ``action_stats.names`` / ``state_stats.names`` in the
checkpoint's ``norm_stats.json`` for tag ``so100_so101_molmoact2``:
  rig 7-dim:    [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                 wrist_flex.pos,   wrist_roll.pos,    gripper.pos,
                 slider.vel]
  policy 6-dim: drop slider.vel.

Action conversion (policy → rig):
  policy 6-dim: as above, no slider.
  rig 7-dim:    append slider.vel = 0.0.

Camera mapping (rig → policy). The SO-100/101 norm_stats has
``camera_keys = []`` and the model card says image order is flexible.
Neither of our cameras is an exact match for the original Realsense
top/side, this is the closest mapping we have:
  overhead_camera → first image (treated as "top")
  arm_camera      → second image (treated as "side")

Inference cadence: the released SO-100/101 checkpoint produces 30-step
action chunks (action_horizon=30 in norm_stats), so at PORTAL_FPS=30
each forward pass buys ~1s of execution. On an H100, a continuous
inference pass with CUDA graphs is documented at ~12GB peak in bf16.
We run one forward pass, execute all 30 steps one tick at a time, then
replan. No temporal ensembling. There is no local engage/disengage —
the operator side gates control via ``c`` to cycle active operators,
exactly as ``policies/act/inference.py``.

Notable MolmoAct2 quirks (from the lerobot port at
``tmp/molmoact2/lerobot/src/lerobot/policies/molmoact2/``):

  * ``norm_tag`` is the single source of truth for direct HF inference
    on a released checkpoint. For SO-100/101 it pins setup_type
    ("single so100/so101 robotic arm in molmoact2"), control_mode
    ("absolute joint pose"), action_horizon=30, n_action_steps=30, and
    the per-dim quantile stats (q01/q99 with all 6 dims masked-in,
    gripper included — unlike Franka/LIBERO which mask the gripper).
  * Internal max_action_dim is fixed at 32; ``predict_action`` slices
    down to the real action dim (6 here) before returning ``out.actions``.
  * ``enable_depth_reasoning`` only applies to the MolmoAct2-Think
    variant. Always pass False for the plain SO-100/101 checkpoint.
  * ``enable_cuda_graph`` caches the inference graph after a couple of
    warmup forward passes; subsequent passes are much faster.
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
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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

IDENTITY = "run-molmo-operator"

def _resolve_config_path() -> Path:
    override = os.environ.get("PORTAL_CONFIG")
    if override:
        return Path(override)
    for candidate in (ROOT / "portal.yaml", ROOT.parent.parent / "portal.yaml"):
        if candidate.exists():
            return candidate
    return ROOT.parent.parent / "portal.yaml"

CONFIG_PATH = _resolve_config_path()

# 6-DOF SO-101 state/action expected by MolmoAct2-SO100_101, in the
# follower's native order (matches the SO-100/101 dataset features used
# during fine-tuning). Slider is intentionally absent here — it is
# handled separately as a passthrough.
POLICY_STATE_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
POLICY_ACTION_KEYS = POLICY_STATE_KEYS

# Per-joint sign + offset to bridge the rig's calibration to the
# SO-100/SO-101 dataset convention the policy was trained in. These are
# the canonical LeRobot v3.0 → v2.1 values documented at
# https://huggingface.co/docs/lerobot/backwardcomp — sign flip on
# shoulder_lift, +90° on shoulder_lift and elbow_flex.
# Applied on policy input (state) and inverted on policy output (action)
# so the closed loop stays consistent:
#   policy_state = signs * rig_state    + offsets
#   rig_action   = (policy_action - offsets) * signs
# Both arrays are aligned with POLICY_STATE_KEYS.
JOINT_SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
JOINT_OFFSETS_DEG = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Slider passthrough. The MolmoAct2 checkpoint has no notion of the
# slider; we always emit zero velocity so the base parks under autonomy.
SLIDER_KEY = "slider.vel"
SLIDER_DEFAULT_VEL = 0.0

# Rig-side action order must match portal.yaml — values flow over the
# wire as a dict but we keep a canonical order for logging.
RIG_ACTION_KEYS = POLICY_ACTION_KEYS + (SLIDER_KEY,)

# Camera mapping rig → policy. The model expects two RGB views; per the
# model card image order doesn't matter. We pass overhead as the first
# image and arm as the second.
TOP_CAMERA = "overhead_camera"
SIDE_CAMERA = "arm_camera"

# norm_tag for SO-100/SO-101 MolmoAct2 normalization stats baked into
# the released checkpoint. Documented on the HF model card.
NORM_TAG = "so100_so101_molmoact2"

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MolmoAct2 ships custom CUDA-graph code paths; MPS won't run it. We
    # fall back to CPU only so the script doesn't crash on a dev laptop,
    # but inference at any usable rate requires a real GPU.
    return "cpu"


def _parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _build_policy_state(state_typed: dict) -> np.ndarray:
    """Strip the slider field and return a 6-dim float32 numpy state in
    the dataset's calibration convention (signs * rig + offsets)."""
    rig = np.array(
        [float(state_typed[k]) for k in POLICY_STATE_KEYS],
        dtype=np.float32,
    )
    return JOINT_SIGNS * rig + JOINT_OFFSETS_DEG


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an RGB uint8 ndarray to a PIL Image. The follower already
    publishes RGB (OpenCVCameraConfig defaults to color_mode=RGB and
    ``frame_bytes_to_numpy_rgb`` keeps it that way), so no channel swap."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(frame))


def _action_dict_from_vec(action_vec) -> dict[str, float]:
    """Build a rig-side 7-dim action dict from a 6-dim policy action,
    inverting the frame transform (action - offsets) * signs and
    pinning slider velocity to zero."""
    a = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if a.shape[0] != len(POLICY_ACTION_KEYS):
        raise ValueError(
            f"expected {len(POLICY_ACTION_KEYS)}-dim policy action, got {a.shape[0]}"
        )
    rig = (a - JOINT_OFFSETS_DEG) * JOINT_SIGNS
    out = {k: float(rig[i]) for i, k in enumerate(POLICY_ACTION_KEYS)}
    out[SLIDER_KEY] = SLIDER_DEFAULT_VEL
    return out


class MolmoActChunkRunner:
    """Owns the latest action chunk and an index into it. Replans when
    the chunk is exhausted (or when ``reset()`` is called).

    The HF checkpoint exposes a single ``predict_action(...)`` entry
    point that runs a flow-matching solver and returns a chunk of
    actions, shape ``(chunk_size, action_dim)`` (also accepts a leading
    batch dim, which we squeeze). Each tick we hand out one row and
    advance the index.
    """

    def __init__(
        self,
        model,
        processor,
        *,
        num_steps: int,
        normalize_language: bool,
        enable_cuda_graph: bool,
    ) -> None:
        self._model = model
        self._processor = processor
        self._num_steps = num_steps
        self._normalize_language = normalize_language
        self._enable_cuda_graph = enable_cuda_graph
        self._chunk: Optional[np.ndarray] = None
        self._idx: int = 0

    def reset(self) -> None:
        self._chunk = None
        self._idx = 0

    @property
    def chunk_size(self) -> int:
        return 0 if self._chunk is None else self._chunk.shape[0]

    def _predict_chunk(
        self,
        *,
        top_rgb: Image.Image,
        side_rgb: Image.Image,
        task: str,
        state_6: np.ndarray,
    ) -> np.ndarray:
        out = self._model.predict_action(
            processor=self._processor,
            images=[top_rgb, side_rgb],
            task=task,
            state=state_6,
            norm_tag=NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=self._num_steps,
            normalize_language=self._normalize_language,
            enable_cuda_graph=self._enable_cuda_graph,
        )
        actions = out.actions
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to("cpu").float().numpy()
        else:
            actions = np.asarray(actions, dtype=np.float32)
        # Squeeze a leading batch dim if present.
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                f"unexpected action chunk shape from predict_action: {actions.shape}"
            )
        return actions

    def step(
        self,
        *,
        top_rgb: Image.Image,
        side_rgb: Image.Image,
        task: str,
        state_6: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Return ``(action_6, replanned)`` for this tick."""
        replanned = False
        if self._chunk is None or self._idx >= self._chunk.shape[0]:
            self._chunk = self._predict_chunk(
                top_rgb=top_rgb,
                side_rgb=side_rgb,
                task=task,
                state_6=state_6,
            )
            self._idx = 0
            replanned = True
            state_pairs = ", ".join(
                f"{k}={float(v):+.2f}" for k, v in zip(POLICY_STATE_KEYS, state_6)
            )
            first = self._chunk[0]
            action_pairs = ", ".join(
                f"{k}={float(v):+.2f}" for k, v in zip(POLICY_ACTION_KEYS, first)
            )
            print(f"[run-molmo-operator] state_in : {state_pairs}")
            print(f"[run-molmo-operator] chunk[0] : {action_pairs}")
        action = self._chunk[self._idx]
        self._idx += 1
        return action, replanned


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MolmoAct2 inference over LiveKit Portal.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MOLMO_MODEL", "allenai/MolmoAct2-SO100_101"),
        help="HF repo id or local path of the MolmoAct2 checkpoint "
             "(env: MOLMO_MODEL).",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "float16"),
        default=os.environ.get("MOLMO_DTYPE", "bfloat16"),
        help="Model dtype. bfloat16 ~16GB VRAM, float32 ~26GB (HF docs).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("MOLMO_NUM_STEPS", "10")),
        help="Continuous flow-matching solver steps per chunk (default 10).",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable CUDA-graph caching (slower steady-state, but no warmup).",
    )
    parser.add_argument(
        "--no-normalize-language",
        dest="normalize_language",
        action="store_false",
        help="Skip the task-string lowercase/trim-punct normalization.",
    )
    parser.set_defaults(cuda_graph=True, normalize_language=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("MOLMO_WARMUP", "2")),
        help="Forward passes to run before connecting to LiveKit so the "
             "first real chunk doesn't stall the room.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MOLMO_TIMEOUT", "60")),
        help="Seconds to drive the robot before the run_molmo() RPC returns "
             "and the operator parks. Set to 0 to disable (RPC returns "
             "immediately after engaging, pre-timeout behaviour). Default 60s.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)
    fps = env_int("PORTAL_FPS", 30)
    device = _select_device()
    dtype = _parse_dtype(args.dtype)
    task_prompt = env_str("LESLIDER_TASK", "Pick the blue cube up and hold it")

    # CUDA-graph caching needs CUDA. Force it off elsewhere so we don't
    # blow up the first forward pass on a CPU dev box.
    enable_cuda_graph = bool(args.cuda_graph and device == "cuda")
    if args.cuda_graph and not enable_cuda_graph:
        print("[run-molmo-operator] CUDA not available; disabling enable_cuda_graph")

    print(
        f"[run-molmo-operator] device={device} dtype={args.dtype} model={args.model} "
        f"num_steps={args.num_steps} cuda_graph={enable_cuda_graph}"
    )

    print("[run-molmo-operator] loading processor ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print("[run-molmo-operator] loading model ... (this can take a few minutes)")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device).eval()

    runner = MolmoActChunkRunner(
        model,
        processor,
        num_steps=args.num_steps,
        normalize_language=args.normalize_language,
        enable_cuda_graph=enable_cuda_graph,
    )

    if args.warmup > 0:
        # The first call (or two, with CUDA graphs) is dramatically
        # slower than steady-state. Bang it out on a dummy obs before
        # we're on the clock so the first real tick doesn't stall.
        print(f"[run-molmo-operator] warmup ({args.warmup} forward passes) ...")
        dummy_state = np.zeros(len(POLICY_STATE_KEYS), dtype=np.float32)
        dummy_top = Image.new("RGB", (224, 224), color=(0, 0, 0))
        dummy_side = Image.new("RGB", (224, 224), color=(0, 0, 0))
        for i in range(args.warmup):
            t0 = time.perf_counter()
            runner._predict_chunk(
                top_rgb=dummy_top, side_rgb=dummy_side,
                task=task_prompt, state_6=dummy_state,
            )
            print(f"[run-molmo-operator] warmup pass {i+1}/{args.warmup}: "
                  f"{(time.perf_counter() - t0) * 1000:.0f} ms")
        runner.reset()

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    latest_obs: Optional[Observation] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs

    op.on_observation(on_observation)
    op.on_active_operator_changed(
        lambda i: print(f"[run-molmo-operator] active operator now: {i}")
    )

    # Mutable prompt shared with the run_molmo RPC. Reads/writes are
    # single Python-level rebinds, so no lock is needed on CPython.
    current_task = task_prompt

    # When a run_molmo() RPC engages the operator with a positive
    # timeout, we hand it an asyncio.Future that resolves once the
    # timeout fires (or a newer RPC preempts it). All writes happen on
    # the asyncio loop so no lock is needed on CPython.
    done_future: Optional[asyncio.Future] = None
    timeout_task: Optional[asyncio.Task] = None

    async def _park(result: dict) -> None:
        nonlocal done_future, timeout_task
        if done_future is None:
            return
        fut = done_future
        done_future = None
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        timeout_task = None
        if not fut.done():
            fut.set_result(result)
        try:
            await op.set_active_operator(None)
        except Exception as e:
            print(f"[run-molmo-operator] release active-operator failed: {e}")

    async def _timeout_watchdog(
        seconds: float, prompt: str, target_future: asyncio.Future
    ) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        if target_future.done() or done_future is not target_future:
            return
        print(
            f"[run-molmo-operator] timeout {seconds:.2f}s elapsed for "
            f"'{prompt}'; parking"
        )
        await _park({
            "status": "done",
            "reason": "timeout",
            "seconds": seconds,
        })

    async def run_molmo(data: RpcInvocationData) -> str:
        # Accept either a bare string ("pick up the blue cube") or a
        # JSON object ({"prompt": "..."}). The continuous tick loop
        # keeps warm-inferencing; we just retarget it and grab the
        # active-operator gate so emitted actions get through.
        nonlocal current_task, done_future, timeout_task
        payload = (data.payload or "").strip()
        new_prompt = payload
        if payload.startswith("{"):
            try:
                new_prompt = str(json.loads(payload).get("prompt", "")).strip()
            except json.JSONDecodeError:
                raise RpcError.Error(
                    code=1400,
                    message="payload was not valid JSON; pass a bare string instead",
                    data=None,
                )
        if not new_prompt:
            raise RpcError.Error(
                code=1400, message="empty prompt", data=None
            )

        current_task = new_prompt
        # Drop the in-flight chunk so the new prompt takes effect on
        # the next tick instead of waiting out the current ~1s chunk.
        runner.reset()

        # Preempt any in-flight run_molmo() so its caller gets a clean
        # reply rather than hanging on a future we're about to drop.
        if done_future is not None and not done_future.done():
            done_future.set_result({"status": "preempted", "prompt": new_prompt})
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        timeout_task = None
        done_future = None

        me_id = op.local_identity()
        await op.set_active_operator(me_id)
        print(
            f"[run-molmo-operator] run_molmo('{new_prompt}') -> "
            f"active operator -> {me_id} timeout={args.timeout}s"
        )

        if args.timeout <= 0:
            return json.dumps({
                "prompt": new_prompt,
                "active_operator": me_id,
                "status": "no_timeout",
            })

        new_future = asyncio.get_running_loop().create_future()
        done_future = new_future
        timeout_task = asyncio.create_task(
            _timeout_watchdog(args.timeout, new_prompt, new_future)
        )

        try:
            result = await new_future
        except asyncio.CancelledError:
            if done_future is new_future:
                await _park({"status": "cancelled"})
            raise

        return json.dumps({
            "prompt": new_prompt,
            "active_operator": me_id,
            **result,
        })

    op.register_rpc_method("run_molmo", run_molmo)

    print(f"[run-molmo-operator] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    me = op.local_identity()

    print(f"[run-molmo-operator] '{IDENTITY}' in '{room}' @ {fps} fps as '{me}' "
          "(teleop drives handoff via `c`)")

    obs_keys_logged = False
    empty_obs_count = 0
    replan_count = 0
    replan_ms_sum = 0.0
    tick_count = 0

    try:
        async for _ in pace(fps):
            obs = latest_obs
            if obs is None:
                empty_obs_count += 1
                if empty_obs_count in (30, 150, 600):
                    print(
                        f"[run-molmo-operator] no observations for {empty_obs_count // 30}s, "
                        "is robot.py running and connected to the same room?"
                    )
                continue
            empty_obs_count = 0

            state_typed = dict(obs.state)
            top_vf = obs.frames.get(TOP_CAMERA)
            side_vf = obs.frames.get(SIDE_CAMERA)
            if top_vf is None or side_vf is None:
                # Both cameras are required for this policy. Skip the
                # tick rather than feeding it a stale or wrong-shape
                # input.
                if tick_count == 0:
                    print(
                        "[run-molmo-operator] waiting for both cameras "
                        f"({TOP_CAMERA}, {SIDE_CAMERA}) ..."
                    )
                continue

            top_rgb = _frame_to_pil(
                frame_bytes_to_numpy_rgb(top_vf.data, top_vf.width, top_vf.height)
            )
            side_rgb = _frame_to_pil(
                frame_bytes_to_numpy_rgb(side_vf.data, side_vf.width, side_vf.height)
            )
            state_6 = _build_policy_state(state_typed)

            if not obs_keys_logged:
                obs_keys_logged = True
                print(
                    f"[run-molmo-operator] first obs: state_keys={sorted(state_typed)} "
                    f"top={TOP_CAMERA} side={SIDE_CAMERA} "
                    f"policy_state_dim={state_6.shape[0]}"
                )

            try:
                t_inf = time.perf_counter()
                action_vec, replanned = runner.step(
                    top_rgb=top_rgb,
                    side_rgb=side_rgb,
                    task=current_task,
                    state_6=state_6,
                )
                inf_ms = (time.perf_counter() - t_inf) * 1000
            except Exception as e:
                print(f"[run-molmo-operator] inference error: {e}")
                continue

            tick_count += 1
            if replanned:
                replan_count += 1
                # Skip the cold-start pass (post-warmup the first call
                # may still graph-record).
                if replan_count > 1:
                    replan_ms_sum += inf_ms
                print(
                    f"[run-molmo-operator] replan #{replan_count}: "
                    f"chunk_size={runner.chunk_size} forward={inf_ms:.0f}ms"
                )

            if tick_count % fps == 0:
                steady_n = max(replan_count - 1, 0)
                if steady_n > 0:
                    avg = replan_ms_sum / steady_n
                    budget_ms = (1000 / fps) * runner.chunk_size
                    marker = "OK" if avg < budget_ms else "OVER"
                    print(
                        f"[run-molmo-operator] tick {tick_count:>5}: replan "
                        f"avg={avg:6.1f}ms (n={steady_n}) "
                        f"chunk_budget={budget_ms:.0f}ms [{marker}]"
                    )

            action = _action_dict_from_vec(action_vec)
            # When the human is the active operator, send_action is
            # dropped at the gate — same convention as ACT.
            op.send_action(
                action,
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us,
            )

    except KeyboardInterrupt:
        print("\n[run-molmo-operator] stopping ...")
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
