"""Hardware builders for the leslider rig.

Concentrates everything that's about *the robot*, serial ports, cameras,
torque caps, so ``robot.py`` and ``teleoperator.py`` can stay focused on
the Portal-side flow. A single arm-mounted camera is assumed.
"""
from __future__ import annotations

import platform

import numpy as np

from lerobot.cameras import Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot_robot_so101_slider import SO101SliderFollower, SO101SliderFollowerConfig
from lerobot_teleoperator_so101_with_slider import (
    SO101WithSliderLeader,
    SO101WithSliderLeaderConfig,
)

from _common import env_bool, env_camera_id, env_int, env_str, required_env

CAMERAS: tuple[str, ...] = ("arm_camera",)
"""Track names. Must match what's declared in ``portal.yaml``; the YAML
is the source of truth for the wire contract. This constant just lets
robot.py and teleoperator.py iterate without reaching back into the
config."""


def build_follower(fps: int) -> SO101SliderFollower:
    """Construct an :class:`SO101SliderFollower` from environment vars.

    Hard-codes ``read_current=False`` to match the lean wire schema in
    ``portal.yaml``. Add the ``*.current`` fields to both the YAML and
    flip this on if you want torque feedback in the dataset.
    """
    width = env_int("LESLIDER_CAM_WIDTH", 640)
    height = env_int("LESLIDER_CAM_HEIGHT", 480)
    cam_defaults = {
        "arm_camera": ("LESLIDER_CAM_ARM", "/dev/video0"),
    }
    # Force V4L2 on Linux. lerobot defaults to CAP_ANY, and OpenCV's auto-
    # picker can land on a backend (often GStreamer) that opens the device
    # but no-ops S_FOURCC/S_FMT — manifests as fourcc='' and a 1920-wide
    # frame that refuses to drop to 640. AVFoundation stays auto-picked on
    # macOS because Cv2Backends.V4L2 is Linux-only.
    backend = Cv2Backends.V4L2 if platform.system() == "Linux" else Cv2Backends.ANY
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=env_camera_id(env_var, default),
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
            backend=backend,
        )
        for name, (env_var, default) in cam_defaults.items()
    }
    return SO101SliderFollower(SO101SliderFollowerConfig(
        id=env_str("LESLIDER_ID", "leslider"),
        port=env_str("LESLIDER_PORT", "/dev/ttyACM0"),
        cameras=cameras,
        read_current=False,
    ))


def build_leader() -> SO101WithSliderLeader:
    """Construct the SO-101 leader arm + slider keyboard handler. The
    leader manages its own pynput listener for arrow keys (slider drive)
    and only exposes a synchronous ``get_action()`` that returns one
    action dict per call."""
    return SO101WithSliderLeader(SO101WithSliderLeaderConfig(
        id=env_str("SO101_LEADER_ID", "so101_leader"),
        port=required_env("SO101_LEADER_PORT"),
        cruise_velocity=env_int("LESLIDER_CRUISE_VELOCITY", 1500),
        speed_increment=env_int("LESLIDER_SPEED_INCREMENT", 250),
        min_velocity=env_int("LESLIDER_MIN_VELOCITY", 100),
        max_velocity=env_int("LESLIDER_MAX_VELOCITY", 3000),
        invert_direction=env_bool("LESLIDER_INVERT_SLIDER", default=False),
    ))


def split_state_frames(observation: dict) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Split a follower observation dict into (state, frames).

    The follower's ``get_observation()`` returns one flat dict mixing
    motor scalars (``shoulder_pan.pos`` etc.) and camera ndarrays. Portal
    wants them on separate channels. ``OpenCVCameraConfig`` defaults
    ``color_mode=RGB``, so frames arrive in the layout Portal expects.
    """
    state = {
        k: float(v) for k, v in observation.items()
        if k.endswith(".pos") or k.endswith(".vel")
    }
    frames = {
        cam: observation[cam]
        for cam in CAMERAS
        if cam in observation and observation[cam] is not None
    }
    return state, frames
