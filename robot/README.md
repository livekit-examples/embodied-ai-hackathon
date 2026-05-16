# portal-robot-runner

Standalone robot-side runtime for the leslider SO-101 follower. This folder is
intentionally robot-only: no data collection, no policies, no training, and no
teleoperator code.

## Setup

```bash
cp .env.example .env
# Fill in LIVEKIT_* and LESLIDER_* values.
uv sync
```

The first run may ask for LeRobot calibration if this follower has not been
calibrated on the machine yet. Calibration files are stored under
`~/.cache/huggingface/lerobot/calibration/`, keyed by `LESLIDER_ID`.

## Run

```bash
uv run robot.py
```

The robot process:

- connects the physical leslider follower and both cameras
- joins the configured LiveKit room as `robot`
- publishes state and video frames
- applies the latest incoming Portal action
- clears the active operator if that peer disconnects, so the arm does not keep
  holding a stale command

## Configuration

`portal.yaml` defines the wire schema: two MJPEG camera streams, seven state
fields, and seven matching action fields. `PORTAL_FPS` can override the YAML fps
without editing the contract.

Useful hardware environment variables:

- `LESLIDER_PORT`: Feetech bus serial port, default `/dev/ttyACM0`
- `LESLIDER_ID`: calibration id, default `leslider`
- `LESLIDER_CAM_ARM`: arm camera device, default `/dev/video0`
- `LESLIDER_CAM_OVERHEAD`: overhead camera device, default `/dev/video2`
- `LESLIDER_CAM_WIDTH` / `LESLIDER_CAM_HEIGHT`: camera resolution
