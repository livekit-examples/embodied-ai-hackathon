"""Hardware builder and observation shaping for the leslider follower."""
from __future__ import annotations

import platform

import numpy as np
from lerobot.cameras import Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot_robot_so101_slider import SO101SliderFollower, SO101SliderFollowerConfig

from _common import env_camera_id, env_int, env_str

CAMERAS: tuple[str, ...] = ("arm_camera", "overhead_camera")


def build_follower(fps: int) -> SO101SliderFollower:
    """Construct an SO-101 slider follower from environment variables."""
    width = env_int("LESLIDER_CAM_WIDTH", 640)
    height = env_int("LESLIDER_CAM_HEIGHT", 480)
    cam_defaults = {
        "arm_camera": ("LESLIDER_CAM_ARM", "/dev/video0"),
        "overhead_camera": ("LESLIDER_CAM_OVERHEAD", "/dev/video2"),
    }
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
    return SO101SliderFollower(
        SO101SliderFollowerConfig(
            id=env_str("LESLIDER_ID", "leslider"),
            port=env_str("LESLIDER_PORT", "/dev/ttyACM0"),
            cameras=cameras,
            read_current=False,
        )
    )


def split_state_frames(
    observation: dict,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Split one follower observation into Portal state and video frames."""
    state = {
        key: float(value)
        for key, value in observation.items()
        if key.endswith(".pos") or key.endswith(".vel")
    }
    frames = {
        camera: observation[camera]
        for camera in CAMERAS
        if camera in observation and observation[camera] is not None
    }
    return state, frames
