#!/usr/bin/env python3
"""
DROID-format ROS2 data collector (15 Hz).

Subscribes to Isaac Sim ROS2 topics, synchronizes via ApproximateTimeSynchronizer,
and records to DROID-compatible HDF5 at 15 Hz.

Usage:
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    python3 droid_sim/scripts/ros2_data_collector.py \
        --task "put the cube in the bowl" --output_dir ./droid_data

Controls (keyboard in this terminal):
    s = start/stop recording
    n = save episode as success, start new
    f = save episode as failure, start new
    q = quit
"""
import argparse
import sys
import os
import time
import threading
from pathlib import Path

import numpy as np
import cv2
import h5py
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import message_filters


class DroidDataCollector(Node):
    """ROS2 node that synchronizes sim data and records at 15 Hz."""

    def __init__(self, task: str, output_dir: str, control_hz: float = 15.0):
        super().__init__("droid_data_collector")
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.control_hz = control_hz
        self.bridge = CvBridge()

        # State
        self.recording = False
        self.episode_count = 0
        self.timesteps = []
        self.last_record_time = 0.0

        # Latest data (updated by subscribers)
        self._lock = threading.Lock()
        self._latest_joints = None
        self._latest_cam_ext1 = None
        self._latest_cam_ext2 = None
        self._latest_cam_wrist = None
        self._latest_tf = {}  # frame_id -> {pos: [x,y,z], quat: [x,y,z,w]}

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers (individual, latest-value pattern)
        self.create_subscription(JointState, "/isaac_joint_states", self._joint_cb, qos)
        self.create_subscription(Image, "/cam_ext1/rgb", self._cam_ext1_cb, qos)
        self.create_subscription(Image, "/cam_ext2/rgb", self._cam_ext2_cb, qos)
        self.create_subscription(Image, "/cam_wrist/rgb", self._cam_wrist_cb, qos)
        self.create_subscription(TFMessage, "/tf", self._tf_cb, qos)

        # Command topic for remote control (start/stop/success/failure)
        from std_msgs.msg import String
        self.create_subscription(String, "/collector/cmd", self._cmd_cb, 10)

        # 15 Hz timer for recording
        self.create_timer(1.0 / self.control_hz, self._record_timer_cb)

        self.get_logger().info(f"DROID Data Collector started ({control_hz} Hz)")
        self.get_logger().info(f"Output: {self.output_dir}")
        self.get_logger().info("Press 's' to start/stop, 'n' for success, 'f' for failure, 'q' to quit")

    # ── Subscriber callbacks ──
    def _joint_cb(self, msg: JointState):
        with self._lock:
            self._latest_joints = msg

    def _cam_ext1_cb(self, msg: Image):
        with self._lock:
            self._latest_cam_ext1 = msg

    def _cam_ext2_cb(self, msg: Image):
        with self._lock:
            self._latest_cam_ext2 = msg

    def _cam_wrist_cb(self, msg: Image):
        with self._lock:
            self._latest_cam_wrist = msg

    def _tf_cb(self, msg: TFMessage):
        """Extract object poses from TF."""
        with self._lock:
            for t in msg.transforms:
                frame = t.child_frame_id
                # Skip robot links (only record scene objects)
                if "panda_" in frame or "finger" in frame or "knuckle" in frame:
                    continue
                self._latest_tf[frame] = {
                    "pos": [
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                    ],
                    "quat": [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ],
                }

    def _cmd_cb(self, msg):
        """Remote control via /collector/cmd topic."""
        cmd = msg.data.strip().lower()
        if cmd == "start":
            self.start_recording()
        elif cmd == "stop":
            self.stop_recording()
        elif cmd == "success":
            self.save_episode(success=True)
        elif cmd == "failure":
            self.save_episode(success=False)
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    # ── 15 Hz recording timer ──
    def _record_timer_cb(self):
        if not self.recording:
            return

        with self._lock:
            joints = self._latest_joints
            ext1 = self._latest_cam_ext1
            ext2 = self._latest_cam_ext2
            wrist = self._latest_cam_wrist
            object_poses = dict(self._latest_tf)  # snapshot

        if joints is None:
            return  # no data yet

        # Extract joint data
        names = list(joints.name)
        positions = np.array(joints.position)
        velocities = np.array(joints.velocity) if joints.velocity else np.zeros_like(positions)
        efforts = np.array(joints.effort) if joints.effort else np.zeros_like(positions)

        # Extract arm joints (panda_joint1-7)
        arm_names = [f"panda_joint{i}" for i in range(1, 8)]
        arm_idx = [names.index(n) for n in arm_names if n in names]
        joint_pos = positions[arm_idx] if arm_idx else np.zeros(7)
        joint_vel = velocities[arm_idx] if arm_idx else np.zeros(7)

        # Gripper
        gripper_val = 0.0
        if "finger_joint" in names:
            gi = names.index("finger_joint")
            gripper_val = positions[gi] / (np.pi / 4)  # normalize to [0,1]

        # Images
        img_ext1 = self._decode_image(ext1)
        img_ext2 = self._decode_image(ext2)
        img_wrist = self._decode_image(wrist)

        self.timesteps.append({
            "timestamp": time.time(),
            "joint_positions": joint_pos.copy(),
            "joint_velocities": joint_vel.copy(),
            "gripper_position": float(gripper_val),
            "efforts": efforts.copy(),
            "exterior_image_1_left": img_ext1,
            "exterior_image_2_left": img_ext2,
            "wrist_image_left": img_wrist,
            "object_poses": object_poses,  # {frame_id: {pos, quat}}
        })

        if len(self.timesteps) % 30 == 0:
            self.get_logger().info(f"Recording... {len(self.timesteps)} steps ({len(self.timesteps)/self.control_hz:.1f}s)")

    def _decode_image(self, msg: Image, target_size=(180, 320)) -> np.ndarray:
        """Convert ROS2 Image to uint8 numpy array."""
        if msg is None:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]))
            return img.astype(np.uint8)
        except Exception as e:
            self.get_logger().warn(f"Image decode error: {e}")
            return np.zeros((*target_size, 3), dtype=np.uint8)

    # ── Episode management ──
    def start_recording(self):
        if self.recording:
            self.get_logger().info("Already recording")
            return
        self.timesteps = []
        self.recording = True
        self.get_logger().info(f">>> Recording episode {self.episode_count} <<<")

    def stop_recording(self):
        self.recording = False
        self.get_logger().info(f"Stopped recording ({len(self.timesteps)} steps)")

    def save_episode(self, success: bool):
        self.stop_recording()
        T = len(self.timesteps)
        if T == 0:
            self.get_logger().warn("Empty episode, skipping")
            return

        ep_dir = self.output_dir / f"episode_{self.episode_count:04d}"
        ep_dir.mkdir(exist_ok=True)

        h5_path = ep_dir / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["success"] = success
            f.attrs["num_steps"] = T
            f.attrs["task_description"] = self.task
            f.attrs["control_frequency_hz"] = self.control_hz

            # Observations
            obs = f.create_group("observation/robot_state")
            obs.create_dataset("joint_positions",
                data=np.array([t["joint_positions"] for t in self.timesteps]))
            obs.create_dataset("joint_velocities",
                data=np.array([t["joint_velocities"] for t in self.timesteps]))
            obs.create_dataset("gripper_position",
                data=np.array([t["gripper_position"] for t in self.timesteps]))

            # Camera images
            cam = f.create_group("observation/camera_images")
            for key in ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]:
                cam.create_dataset(key,
                    data=np.array([t[key] for t in self.timesteps]),
                    dtype=np.uint8, compression="gzip", compression_opts=4)

            # Actions (current joint positions as action proxy)
            act = f.create_group("action")
            act.create_dataset("joint_position",
                data=np.array([t["joint_positions"] for t in self.timesteps]))
            act.create_dataset("gripper_position",
                data=np.array([t["gripper_position"] for t in self.timesteps]))

            # Timestamps
            f.create_dataset("timestamps",
                data=np.array([t["timestamp"] for t in self.timesteps]))

            # Object poses (all scene objects from TF)
            all_obj_names = set()
            for t in self.timesteps:
                all_obj_names.update(t["object_poses"].keys())
            if all_obj_names:
                obj_grp = f.create_group("observation/object_poses")
                for obj_name in sorted(all_obj_names):
                    positions = []
                    orientations = []
                    for t in self.timesteps:
                        pose = t["object_poses"].get(obj_name, {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]})
                        positions.append(pose["pos"])
                        orientations.append(pose["quat"])
                    safe_name = obj_name.replace("/", "_").strip("_")
                    obj_grp.create_dataset(f"{safe_name}/position", data=np.array(positions, dtype=np.float64))
                    obj_grp.create_dataset(f"{safe_name}/orientation", data=np.array(orientations, dtype=np.float64))

        # Metadata
        meta = {
            "task_description": self.task,
            "episode_id": self.episode_count,
            "num_steps": T,
            "success": success,
            "control_frequency_hz": self.control_hz,
            "duration_s": T / self.control_hz,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f"Saved episode {self.episode_count}: {h5_path} "
            f"({T} steps, {T/self.control_hz:.1f}s, {'SUCCESS' if success else 'FAILURE'})"
        )
        self.episode_count += 1
        self.timesteps = []


def keyboard_thread(collector: DroidDataCollector):
    """Handle keyboard input in separate thread."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while rclpy.ok():
            ch = sys.stdin.read(1)
            if ch == "s":
                if collector.recording:
                    collector.stop_recording()
                else:
                    collector.start_recording()
            elif ch == "n":
                collector.save_episode(success=True)
                print("\r\n[Saved SUCCESS] Press 's' to start next episode\r\n")
            elif ch == "f":
                collector.save_episode(success=False)
                print("\r\n[Saved FAILURE] Press 's' to start next episode\r\n")
            elif ch == "q":
                if collector.recording:
                    collector.save_episode(success=False)
                print("\r\n[Quit]\r\n")
                rclpy.shutdown()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    parser = argparse.ArgumentParser(description="DROID ROS2 Data Collector")
    parser.add_argument("--task", type=str, default="manipulation task")
    parser.add_argument("--output_dir", type=str, default="./droid_data")
    parser.add_argument("--hz", type=float, default=15.0)
    args = parser.parse_args()

    rclpy.init()
    collector = DroidDataCollector(
        task=args.task,
        output_dir=args.output_dir,
        control_hz=args.hz,
    )

    # Keyboard in background thread (only if running in a real terminal)
    import io
    if hasattr(sys.stdin, 'fileno') and not isinstance(sys.stdin, io.StringIO):
        try:
            import termios
            termios.tcgetattr(sys.stdin.fileno())
            kb = threading.Thread(target=keyboard_thread, args=(collector,), daemon=True)
            kb.start()
            has_keyboard = True
        except Exception:
            has_keyboard = False
    else:
        has_keyboard = False

    print("\n" + "=" * 50)
    print("DROID ROS2 Data Collector")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Rate: {args.hz} Hz")
    print(f"Output: {args.output_dir}")
    if has_keyboard:
        print("Controls: s=start/stop  n=success  f=failure  q=quit")
    else:
        print("No TTY — auto-start recording.")
        print("Stop with: ros2 topic pub /collector/cmd std_msgs/String '{data: stop}' --once")
    print("=" * 50 + "\n")

    # Auto-start if no keyboard
    if not has_keyboard:
        collector.start_recording()

    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    finally:
        if collector.recording:
            collector.save_episode(success=False)
        collector.destroy_node()
        rclpy.try_shutdown()
        print(f"\nDone. {collector.episode_count} episodes saved to {args.output_dir}")


if __name__ == "__main__":
    main()
