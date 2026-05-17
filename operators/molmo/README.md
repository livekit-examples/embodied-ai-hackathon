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
uv run python molmo_operator.py
```

The policy joins the configured LiveKit room as `run-molmo-operator`. The teleop side
gates handoff by cycling the active operator with `c`. Molmo predicts the
6-DOF SO-101 arm action; this wrapper appends `slider.vel = 0.0` so the linear
slider parks while the arm runs autonomously.

Useful environment overrides:

```bash
LESLIDER_TASK="Pick the blue cube up and hold it" uv run python molmo_operator.py
MOLMO_NUM_STEPS=20 MOLMO_DTYPE=bfloat16 uv run python molmo_operator.py
```

### `run_molmo` RPC

The operator registers a `run_molmo(prompt)` RPC that retargets the policy
to a new task string and claims the active-operator slot. The reply is
blocked until a timeout fires (default 30s), at which point the operator
parks (releases active-operator) and the RPC returns:

```json
{"prompt": "...", "active_operator": "run-molmo-operator", "status": "done", "reason": "timeout", "seconds": 30.0}
```

A newer `run_molmo` call preempts an in-flight one; the preempted caller
gets `{"status": "preempted", ...}`.

Configure the timeout with `--timeout SECONDS` or `MOLMO_TIMEOUT=SECONDS`.
Pass `--timeout 0` to disable blocking (the RPC returns immediately with
`{"status": "no_timeout", ...}` and the operator keeps driving until a
peer claims the active-operator slot via `c`).

```bash
uv run python molmo_operator.py --timeout 45
MOLMO_TIMEOUT=10 uv run python molmo_operator.py
uv run python molmo_operator.py --timeout 0
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
`uv run python molmo_operator.py`.
