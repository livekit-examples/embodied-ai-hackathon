"""Standalone smoke test for MolmoAct2-SO100_101 inference.

Verifies the box this script is launched on can actually host the
model end-to-end. No LiveKit, no robot. Uses the model card's
reference inputs as the canonical "known good" forward pass:

  * loads model + processor with `transformers`
  * downloads the two sample Realsense frames the model card ships
  * uses the model card's sample 6-DOF state + lemon-bowl task
  * synthesizes a rig-style 7-dim observation (slider.vel=0) and
    runs it through the *real* helpers from `molmo_operator.py`
    (`_build_policy_state`, `MolmoActChunkRunner`,
    `_action_dict_from_vec`), so the slider-strip / chunk consumption
    / slider-passthrough code paths are exercised
  * reports chunk shape, replan timing, peak GPU memory, and a sample
    of rig-side action dicts

This is meant to be the first thing you run on a fresh Nebius cluster
before pointing the policy at a live room. It is intentionally a
single-pass script, not a streaming loop.

Usage:
  uv run python smoke_test.py
  uv run python smoke_test.py --dtype float32 --ticks 60
  uv run python smoke_test.py --model /local/path/to/checkpoint
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Pull the rig-side wiring straight out of the inference module so the
# smoke test stays in lockstep with the production code path. If those
# helpers move or change shape, this test breaks loudly — which is the
# point.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from molmo_operator import (  # noqa: E402
    POLICY_STATE_KEYS,
    RIG_ACTION_KEYS,
    SLIDER_KEY,
    MolmoActChunkRunner,
    _action_dict_from_vec,
    _build_policy_state,
    _parse_dtype,
    _select_device,
)

# Sample inputs from the MolmoAct2-SO100_101 model card. We use them
# verbatim so a smoke-test pass is bit-for-bit comparable to the model
# card's reference example.
SAMPLE_STATE_6DOF = np.array(
    [
        -0.52734375,
        189.140625,
        181.40625,
        60.64453125,
        -3.603515625,
        1.0971786975860596,
    ],
    dtype=np.float32,
)
SAMPLE_TASK = (
    "Move the arm towards the lemon, grasp it, lift it up, "
    "and drop it into the red bowl."
)
SAMPLE_TOP_ASSET = "assets/sample_realsense_top_rgb.png"
SAMPLE_SIDE_ASSET = "assets/sample_realsense_side_rgb.png"


def _build_rig_state_dict(state_6: np.ndarray) -> dict[str, float]:
    """Synthesize the rig-style 7-dim state dict robot.py would publish."""
    rig: dict[str, float] = {
        k: float(state_6[i]) for i, k in enumerate(POLICY_STATE_KEYS)
    }
    rig[SLIDER_KEY] = 0.0
    return rig


def _peak_gpu_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("MOLMO_MODEL", "allenai/MolmoAct2-SO100_101"),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "float16"),
        default=os.environ.get("MOLMO_DTYPE", "bfloat16"),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("MOLMO_NUM_STEPS", "10")),
        help="Flow-matching solver steps per chunk.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=35,
        help="Synthetic ticks to run. With chunk_size=30 this triggers "
             "exactly one replan, which is what we want.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup forward passes before the timed ticks. The first "
             "pass also records the CUDA graph when --no-cuda-graph "
             "is off.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable CUDA-graph caching (slower steady-state, no warmup).",
    )
    parser.add_argument(
        "--no-normalize-language",
        dest="normalize_language",
        action="store_false",
    )
    parser.set_defaults(cuda_graph=True, normalize_language=True)
    args = parser.parse_args()

    device = _select_device()
    dtype = _parse_dtype(args.dtype)
    enable_cuda_graph = bool(args.cuda_graph and device == "cuda")

    print(f"[smoke] device={device} dtype={args.dtype} model={args.model}")
    print(f"[smoke] num_steps={args.num_steps} cuda_graph={enable_cuda_graph} "
          f"warmup={args.warmup} ticks={args.ticks}")
    if device != "cuda":
        print("[smoke] WARNING: no CUDA device. The model effectively requires "
              "an NVIDIA GPU; loading/inference may hang or OOM on CPU.")

    # 1. Sample images from the model card. These live in the model repo
    #    under assets/, so `hf_hub_download` pulls them into the HF cache.
    print(f"[smoke] downloading sample images from {args.model} ...")
    top_path = hf_hub_download(args.model, SAMPLE_TOP_ASSET)
    side_path = hf_hub_download(args.model, SAMPLE_SIDE_ASSET)
    top_rgb = Image.open(top_path).convert("RGB")
    side_rgb = Image.open(side_path).convert("RGB")
    print(f"[smoke] top={top_rgb.size}  side={side_rgb.size}")

    # 2. Load processor + model. Same path molmo_operator.py uses.
    print("[smoke] loading processor ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print("[smoke] loading model (first call downloads the checkpoint) ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device).eval()
    print(f"[smoke] model loaded in {time.perf_counter() - t0:.1f}s")

    # 3. Synthesize a rig-side 7-dim observation, then run it through
    #    the production helpers. This is the bit that proves the
    #    7→6 strip and the 6→7 slider passthrough actually work.
    rig_state = _build_rig_state_dict(SAMPLE_STATE_6DOF)
    policy_state = _build_policy_state(rig_state)
    assert policy_state.shape == (6,), policy_state.shape
    assert np.allclose(policy_state, SAMPLE_STATE_6DOF), (
        "rig→policy state strip changed values, this should be a pure permute"
    )
    print(f"[smoke] rig state (7-dim) → policy state (6-dim) OK: {policy_state}")

    runner = MolmoActChunkRunner(
        model,
        processor,
        num_steps=args.num_steps,
        normalize_language=args.normalize_language,
        enable_cuda_graph=enable_cuda_graph,
    )

    # 4. Warmup. The first pass is dominated by CUDA-graph capture and
    #    kernel autotuning; the second is the first "real" steady-state
    #    measurement.
    print(f"[smoke] warmup x{args.warmup} ...")
    for i in range(args.warmup):
        t = time.perf_counter()
        runner._predict_chunk(
            top_rgb=top_rgb, side_rgb=side_rgb,
            task=SAMPLE_TASK, state_6=policy_state,
        )
        print(f"[smoke]   warmup {i+1}/{args.warmup}: "
              f"{(time.perf_counter() - t) * 1000:.0f} ms")
    runner.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 5. Timed ticks. Index 0 always replans. With chunk_size=30 the
    #    default --ticks=35 forces a second replan at index 30.
    print(f"[smoke] running {args.ticks} synthetic ticks ...")
    replan_ms: list[float] = []
    tick_ms: list[float] = []
    first_action: dict[str, float] | None = None
    last_action: dict[str, float] | None = None
    for i in range(args.ticks):
        t = time.perf_counter()
        action_vec, replanned = runner.step(
            top_rgb=top_rgb, side_rgb=side_rgb,
            task=SAMPLE_TASK, state_6=policy_state,
        )
        elapsed_ms = (time.perf_counter() - t) * 1000
        tick_ms.append(elapsed_ms)
        if replanned:
            replan_ms.append(elapsed_ms)
            print(f"[smoke]   tick {i:>3}: replan, chunk_size={runner.chunk_size} "
                  f"forward={elapsed_ms:.0f} ms")
        action = _action_dict_from_vec(action_vec)
        if first_action is None:
            first_action = action
        last_action = action

    # 6. Assertions: shape, slider passthrough, finite values.
    assert first_action is not None and last_action is not None
    assert set(first_action) == set(RIG_ACTION_KEYS), (
        f"unexpected action keys: {sorted(first_action)} vs {RIG_ACTION_KEYS}"
    )
    assert first_action[SLIDER_KEY] == 0.0, (
        f"slider passthrough should be 0.0, got {first_action[SLIDER_KEY]}"
    )
    for k, v in first_action.items():
        assert np.isfinite(v), f"non-finite action value at {k}: {v}"

    # 7. Report.
    print()
    print("[smoke] === summary ===")
    print(f"[smoke] chunk_size:        {runner.chunk_size}")
    print(f"[smoke] replans:           {len(replan_ms)}")
    if replan_ms:
        print(f"[smoke] replan forward ms: {replan_ms} "
              f"(min={min(replan_ms):.0f}, max={max(replan_ms):.0f})")
    non_replan = [ms for ms, is_r in zip(tick_ms, [False] + [False] * (len(tick_ms) - 1))
                  if not is_r]
    # Easier: ticks not in replan_ms
    replan_set = set(replan_ms)
    non_replan = [ms for ms in tick_ms if ms not in replan_set]
    if non_replan:
        avg = sum(non_replan) / len(non_replan)
        print(f"[smoke] non-replan tick ms avg={avg:.2f} "
              f"(n={len(non_replan)}, max={max(non_replan):.2f})")
    peak = _peak_gpu_memory_gib()
    if peak is not None:
        print(f"[smoke] peak GPU mem:      {peak:.2f} GiB")
    print(f"[smoke] first action:      {first_action}")
    print(f"[smoke] last action:       {last_action}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
