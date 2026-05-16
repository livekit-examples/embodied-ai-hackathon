# leslider Molmo inference

Standalone MolmoAct2-SO100_101 inference operator for the leslider robot.
This is the Molmo policy slice from `portal-hitl`, flattened so it can run
from this directory without the rest of that repo.

## Setup

```bash
cp .env.example .env
# Fill LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_ROOM.
uv sync
```

The robot and teleoperator processes still need to be running in the same
LiveKit room. This project only provides the policy operator.

## Smoke test

Run this first on the GPU machine. It downloads the model-card sample images,
loads `allenai/MolmoAct2-SO100_101`, runs a synthetic action chunk, and checks
the leslider 7-dim to SO-101 6-dim conversion.

```bash
uv run python smoke_test.py
```

Useful overrides:

```bash
MOLMO_DTYPE=float32 uv run python smoke_test.py
MOLMO_NUM_STEPS=20 uv run python smoke_test.py
uv run python smoke_test.py --model /local/path/to/checkpoint
```

## Live inference

```bash
uv run python inference.py
```

The policy joins the configured LiveKit room as `policy-molmo`. The teleop side
gates handoff by cycling the active operator with `c`. Molmo predicts the
6-DOF SO-101 arm action; this wrapper appends `slider.vel = 0.0` so the linear
slider parks while the arm runs autonomously.

Useful environment overrides:

```bash
LESLIDER_TASK="Pick the blue cube up and hold it" uv run python inference.py
MOLMO_NUM_STEPS=20 MOLMO_DTYPE=bfloat16 uv run python inference.py
```

## Cloud

SkyPilot configs are included:

```bash
sky launch -c molmo skypilot/smoke_test.yaml
sky launch -c molmo skypilot/inference.yaml
```

The cluster name is intentionally always `molmo`. The YAML files cannot force
SkyPilot's cluster name by themselves, so the launch command supplies `-c molmo`.
The wrapper scripts do that for you:

```bash
scripts/launch_smoke.sh
scripts/launch_inference.sh
```

For live inference, `skypilot/inference.yaml` mounts `.env` to the cloud VM as
`~/.leslider-molmo.env`, copies it into the workdir at runtime, then runs
`uv run python inference.py`.
