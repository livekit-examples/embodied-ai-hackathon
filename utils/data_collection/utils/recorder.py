"""Off-thread LeRobotDataset recorder for the operator-side HITL loop.

Build with the same field schemas declared in ``portal.yaml`` (state +
action) and the camera resolution received from the robot. Each call to
``push_frame`` adds one timestep to the in-flight episode; ``end_episode``
flushes to disk on a background thread so the operator's tick rate isn't
held up by parquet/video encoding.

Resumes an existing dataset on disk if the ``meta/tasks.parquet`` marker
is present, useful when collecting a long-running session in chunks.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot.utils.constants import ACTION, OBS_STR

logger = logging.getLogger(__name__)


class DatasetRecorder:
    """Wraps `LeRobotDataset` with non-blocking save + start/stop/discard.

    The recorder is a passive pipe: ``push_frame`` is a no-op when not
    recording. ``start_episode`` waits for any prior background save to
    finish first so the dataset's internal episode buffer is clean.
    """

    def __init__(
        self,
        repo_id: str,
        fps: int,
        state_field_names: Sequence[str],
        action_field_names: Sequence[str],
        cameras: Sequence[str],
        camera_height: int,
        camera_width: int,
        task: str,
        *,
        root: str | Path | None = None,
        robot_type: str = "so101_slider_follower",
    ) -> None:
        self._task = task
        self._cameras = tuple(cameras)

        # Mirror the YAML schema into lerobot's hw_features shape: the
        # state fields become named floats; each camera goes in as a
        # (H, W, 3) tuple. Both are routed through hw_to_dataset_features
        # so the resulting dataset matches what a policy expects to ingest.
        obs_hw: dict[str, Any] = {name: float for name in state_field_names}
        for cam in self._cameras:
            obs_hw[cam] = (camera_height, camera_width, 3)
        act_hw: dict[str, Any] = {name: float for name in action_field_names}

        obs_ds = hw_to_dataset_features(obs_hw, OBS_STR, use_video=True)
        act_ds = hw_to_dataset_features(act_hw, ACTION, use_video=True)
        self._ds_features = combine_feature_dicts(obs_ds, act_ds)

        # No HF Hub roundtrips for local recording. Setting OFFLINE here
        # is enough; LeRobotDataset checks it before any network call.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        resolved_root = Path(root) if root is not None else Path("data") / repo_id
        tasks_marker = resolved_root / "meta" / "tasks.parquet"
        if tasks_marker.exists():
            logger.info("resuming dataset at %s (%s)", resolved_root, repo_id)
            self.dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=resolved_root,
                image_writer_processes=0,
                image_writer_threads=4,
                streaming_encoding=True,
                encoder_threads=2,
            )
        else:
            if resolved_root.exists():
                # An aborted prior run can leave a stub directory without
                # the tasks marker. Wipe so `create` doesn't error.
                logger.warning("wiping incomplete dataset at %s", resolved_root)
                shutil.rmtree(resolved_root)
            logger.info("creating dataset at %s (%s)", resolved_root, repo_id)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=self._ds_features,
                root=resolved_root,
                robot_type=robot_type,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4,
                streaming_encoding=True,
                encoder_threads=2,
            )

        self._recording = False
        self._save_thread: threading.Thread | None = None
        self._save_lock = threading.Lock()
        self._episode_count = self.dataset.num_episodes
        self._errors: list[str] = []
        self._errors_lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str | None = None) -> bool:
        if self._recording:
            return False
        self._await_prior_save()
        if task:
            self._task = task
        self._recording = True
        logger.info("episode %d recording (task=%r)", self._episode_count, self._task)
        return True

    def push_frame(self, observation: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        if not self._recording:
            return
        try:
            obs_frame = build_dataset_frame(self._ds_features, dict(observation), prefix=OBS_STR)
            act_frame = build_dataset_frame(self._ds_features, dict(action), prefix=ACTION)
            frame = {**obs_frame, **act_frame, "task": self._task}
            self.dataset.add_frame(frame)
        except Exception as exc:
            logger.exception("push_frame failed")
            self._record_error(f"push_frame: {exc}")
            self._recording = False

    def end_episode(self) -> bool:
        if not self._recording:
            return False
        self._recording = False
        if not self.dataset.has_pending_frames():
            return True
        with self._save_lock:
            self._save_thread = threading.Thread(
                target=self._save_worker,
                name=f"dataset-save-ep{self._episode_count}",
                daemon=True,
            )
            self._save_thread.start()
        self._episode_count += 1
        return True

    def discard_episode(self) -> bool:
        """Drop the in-flight episode buffer. Returns True only when an
        active recording was actually thrown away. Once ``end_episode``
        has handed the buffer to the save thread the episode is already
        committing to disk — saved episodes are not rolled back here, so
        discard becomes a no-op and returns False."""
        if not self._recording:
            return False
        self._recording = False
        if not self.dataset.has_pending_frames():
            return True
        try:
            self.dataset.clear_episode_buffer(delete_images=True)
        except Exception as exc:
            logger.exception("clear_episode_buffer failed")
            self._record_error(f"discard: {exc}")
            return False
        return True

    def finalize(self) -> None:
        self._await_prior_save()
        try:
            self.dataset.finalize()
        except Exception as exc:
            logger.exception("dataset.finalize failed")
            self._record_error(f"finalize: {exc}")

    def poll_errors(self) -> list[str]:
        with self._errors_lock:
            errs = self._errors
            self._errors = []
        return errs

    def _save_worker(self) -> None:
        episode = self._episode_count - 1
        started = time.perf_counter()
        try:
            self.dataset.save_episode()
        except Exception as exc:
            logger.exception("save_episode failed for episode %d", episode)
            self._record_error(f"save_episode(ep={episode}): {exc}")
            return
        logger.info("episode %d saved in %.2fs", episode, time.perf_counter() - started)

    def _record_error(self, message: str) -> None:
        with self._errors_lock:
            self._errors.append(message)

    def _await_prior_save(self) -> None:
        with self._save_lock:
            t = self._save_thread
            self._save_thread = None
        if t is not None:
            t.join()


def build_from_frame(
    *,
    repo_id: str,
    fps: int,
    state_field_names: Sequence[str],
    action_field_names: Sequence[str],
    cameras: Sequence[str],
    frame,
    task: str,
    root: str | Path | None = None,
) -> DatasetRecorder:
    """Construct a :class:`DatasetRecorder` using the resolution
    discovered from a received frame. The wire schema doesn't pin
    width/height, so we pull them off the first frame the robot
    publishes and feed them to the recorder."""
    h, w = frame.shape[:2]
    return DatasetRecorder(
        repo_id=repo_id,
        fps=fps,
        state_field_names=state_field_names,
        action_field_names=action_field_names,
        cameras=cameras,
        camera_height=h,
        camera_width=w,
        task=task,
        root=root,
    )
