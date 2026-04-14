"""DROID-compatible HDF5 trajectory recorder.

Records trajectories in the DROID HDF5 format:
  observation/robot_state/{cartesian_position, gripper_position, joint_positions, joint_velocities}
  observation/camera_images/{exterior_image_1_left, exterior_image_2_left, wrist_image_left}
  action/{cartesian_position, cartesian_velocity, joint_position, joint_velocity, gripper_position, gripper_velocity}

Images are stored per-timestep as uint8 arrays (180x320x3) to keep things simple.
For full DROID compatibility, a post-processing step can convert to MP4.
"""
import os
import time
import json
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np


class DroidRecorder:
    """Records episodes in DROID-compatible HDF5 format."""

    def __init__(self, output_dir: str, image_size: tuple = (180, 320)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size  # (H, W)

        self._episode_count = 0
        self._timesteps = []
        self._recording = False
        self._episode_metadata = {}

    def start_episode(self, task_description: str = "", scene_id: str = "1"):
        """Begin recording a new episode."""
        self._timesteps = []
        self._recording = True
        self._episode_metadata = {
            "task_description": task_description,
            "scene_id": scene_id,
            "start_time": time.time(),
        }
        print(f"[Recorder] Episode {self._episode_count} started: '{task_description}'")

    def record_timestep(
        self,
        joint_positions: np.ndarray,       # (7,)
        joint_velocities: np.ndarray,      # (7,)
        gripper_position: float,           # [0, 1]
        ee_pose: np.ndarray,               # (6,) [x, y, z, roll, pitch, yaw]
        action_joint_pos: np.ndarray,      # (7,)
        action_gripper: float,             # [0, 1]
        exterior_image_1: np.ndarray,      # RGB
        exterior_image_2: np.ndarray,      # RGB
        wrist_image: np.ndarray,           # RGB
        prev_action_joint_pos: Optional[np.ndarray] = None,
    ):
        """Record a single timestep."""
        if not self._recording:
            return

        # Resize images
        img1 = self._resize(exterior_image_1)
        img2 = self._resize(exterior_image_2)
        img_w = self._resize(wrist_image)

        # Compute velocity-based action (delta from previous)
        if prev_action_joint_pos is not None:
            action_joint_vel = action_joint_pos - prev_action_joint_pos
        else:
            action_joint_vel = np.zeros(7)

        # Compute cartesian velocity (simple finite difference)
        if len(self._timesteps) > 0:
            prev_ee = self._timesteps[-1]["observation"]["robot_state"]["cartesian_position"]
            ee_vel = ee_pose - prev_ee
        else:
            ee_vel = np.zeros(6)

        gripper_vel = 0.0
        if len(self._timesteps) > 0:
            prev_g = self._timesteps[-1]["observation"]["robot_state"]["gripper_position"]
            gripper_vel = action_gripper - prev_g

        self._timesteps.append({
            "observation": {
                "robot_state": {
                    "cartesian_position": ee_pose.astype(np.float64),
                    "gripper_position": np.float64(gripper_position),
                    "joint_positions": joint_positions.astype(np.float64),
                    "joint_velocities": joint_velocities.astype(np.float64),
                },
                "camera_images": {
                    "exterior_image_1_left": img1,
                    "exterior_image_2_left": img2,
                    "wrist_image_left": img_w,
                },
            },
            "action": {
                "cartesian_position": ee_pose.astype(np.float64),
                "cartesian_velocity": ee_vel.astype(np.float64),
                "joint_position": action_joint_pos.astype(np.float64),
                "joint_velocity": action_joint_vel.astype(np.float64),
                "gripper_position": np.float64(action_gripper),
                "gripper_velocity": np.float64(gripper_vel),
            },
            "timestamp": time.time(),
        })

    def end_episode(self, success: bool = True) -> str:
        """Finish and save the episode. Returns path to saved HDF5."""
        if not self._recording:
            return ""

        self._recording = False
        T = len(self._timesteps)
        if T == 0:
            print("[Recorder] Empty episode, skipping.")
            return ""

        # Create episode directory
        ep_dir = self.output_dir / f"episode_{self._episode_count:04d}"
        ep_dir.mkdir(exist_ok=True)

        # Save trajectory.h5
        h5_path = ep_dir / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            # Attributes
            f.attrs["success"] = success
            f.attrs["num_steps"] = T
            f.attrs["task_description"] = self._episode_metadata.get("task_description", "")
            f.attrs["scene_id"] = self._episode_metadata.get("scene_id", "")

            # Observation / robot_state
            obs_rs = f.create_group("observation/robot_state")
            obs_rs.create_dataset("cartesian_position",
                data=np.array([t["observation"]["robot_state"]["cartesian_position"] for t in self._timesteps]))
            obs_rs.create_dataset("gripper_position",
                data=np.array([t["observation"]["robot_state"]["gripper_position"] for t in self._timesteps]))
            obs_rs.create_dataset("joint_positions",
                data=np.array([t["observation"]["robot_state"]["joint_positions"] for t in self._timesteps]))
            obs_rs.create_dataset("joint_velocities",
                data=np.array([t["observation"]["robot_state"]["joint_velocities"] for t in self._timesteps]))

            # Observation / camera images
            obs_cam = f.create_group("observation/camera_images")
            obs_cam.create_dataset("exterior_image_1_left",
                data=np.array([t["observation"]["camera_images"]["exterior_image_1_left"] for t in self._timesteps]),
                dtype=np.uint8, compression="gzip", compression_opts=4)
            obs_cam.create_dataset("exterior_image_2_left",
                data=np.array([t["observation"]["camera_images"]["exterior_image_2_left"] for t in self._timesteps]),
                dtype=np.uint8, compression="gzip", compression_opts=4)
            obs_cam.create_dataset("wrist_image_left",
                data=np.array([t["observation"]["camera_images"]["wrist_image_left"] for t in self._timesteps]),
                dtype=np.uint8, compression="gzip", compression_opts=4)

            # Actions
            act = f.create_group("action")
            for key in ["cartesian_position", "cartesian_velocity", "joint_position",
                        "joint_velocity", "gripper_position", "gripper_velocity"]:
                act.create_dataset(key,
                    data=np.array([t["action"][key] for t in self._timesteps]))

            # Flat action (DROID convention: cartesian_position(6) + gripper(1))
            flat_action = np.array([
                np.concatenate([t["action"]["cartesian_position"], [t["action"]["gripper_position"]]])
                for t in self._timesteps
            ])
            f.create_dataset("action_flat", data=flat_action)

        # Save metadata JSON
        meta = {
            **self._episode_metadata,
            "end_time": time.time(),
            "num_steps": T,
            "success": success,
            "control_frequency_hz": 15,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Recorder] Episode {self._episode_count} saved: {h5_path} ({T} steps, success={success})")
        self._episode_count += 1
        self._timesteps = []
        return str(h5_path)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to DROID standard size."""
        if img.shape[:2] != self.image_size:
            return cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        return img
