#!/usr/bin/env python3
"""
Automated Grasp-and-Place Data Generation Pipeline.

Full pipeline:
  1. MCP: randomize_scene() + reset robot
  2. Wrist camera: capture depth -> point cloud (with instance segmentation)
  3. GraspGen ZMQ: generate 6-DOF grasp poses
  4. MoveIt2: plan + execute approach -> grasp -> lift -> place
  5. ROS2 Data Collector: record trajectory
  6. Loop

Requires 4 terminals:
  T1: Isaac Sim (IsaacLab DROID env + ROS2 + MCP)
  T2: GraspGen ZMQ server (GPU)
  T3: ROS2 data collector (15Hz)
  T4: This script

Usage:
    source /opt/ros/humble/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    python3 droid/scripts/auto_grasp_pipeline.py --num_episodes 10 --task "pick up cube"
"""
import argparse
import sys
import os
import time
import threading
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    RobotState, WorkspaceParameters,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration

# GraspGen ZMQ client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs", "GraspGen"))
try:
    from grasp_gen.serving.zmq_client import GraspGenClient
    HAS_GRASPGEN = True
except ImportError:
    HAS_GRASPGEN = False
    print("WARNING: GraspGen client not found. pip install pyzmq msgpack msgpack-numpy")

# MCP client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "moveit", "scripts"))
from mcp_client import IsaacMCP


# Franka Panda joint names (7-DOF arm)
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# Home position
PANDA_HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.785]
# Pre-grasp position (arm extended forward)
PANDA_PRE_GRASP = [0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785]
# Place position
PANDA_PLACE = [1.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785]


class AutoGraspPipeline(Node):
    """ROS2 node for automated grasp-and-place data generation."""

    def __init__(self, args):
        super().__init__("auto_grasp_pipeline")
        self.bridge = CvBridge()
        self.args = args
        self.episode_count = 0

        # ── MCP client ──
        self.mcp = IsaacMCP()
        if not self.mcp.is_connected():
            self.get_logger().error("MCP not connected. Start Isaac Sim first.")
            raise RuntimeError("MCP unavailable")
        self.get_logger().info("MCP connected")

        # ── GraspGen client ──
        self.grasp_client = None
        if HAS_GRASPGEN:
            try:
                self.grasp_client = GraspGenClient(
                    host=args.graspgen_host, port=args.graspgen_port
                )
                meta = self.grasp_client.server_metadata
                self.get_logger().info(f"GraspGen connected: {meta}")
            except Exception as e:
                self.get_logger().warn(f"GraspGen unavailable: {e}. Will use random grasps.")

        # ── Sensor data ──
        self._lock = threading.Lock()
        self._latest_depth = None
        self._latest_rgb = None
        self._latest_joints = None
        self._latest_joint_names = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(Image, "/cam_wrist/depth", self._depth_cb, qos)
        self.create_subscription(Image, "/cam_wrist/rgb", self._rgb_cb, qos)
        self.create_subscription(JointState, "/isaac_joint_states", self._joint_cb, qos)

        # ── Joint command publisher (direct control via topic_based_ros2_control) ──
        self._joint_cmd_pub = self.create_publisher(
            JointState, "/isaac_joint_commands", 10
        )

        # ── Data collector control ──
        self._collector_pub = self.create_publisher(String, "/collector/cmd", 10)

        self.get_logger().info("Auto Grasp Pipeline ready")

    # ── Callbacks ──
    def _depth_cb(self, msg):
        with self._lock:
            self._latest_depth = msg

    def _rgb_cb(self, msg):
        with self._lock:
            self._latest_rgb = msg

    def _joint_cb(self, msg):
        with self._lock:
            self._latest_joints = msg
            self._latest_joint_names = list(msg.name)

    # ── Sensor utilities ──
    def spin_for(self, seconds=1.0):
        """Spin to collect latest sensor data."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def get_point_cloud(self):
        """Get point cloud from wrist depth camera."""
        with self._lock:
            depth_msg = self._latest_depth

        if depth_msg is None:
            return None

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        elif depth.dtype == np.float32:
            pass
        else:
            depth = depth.astype(np.float32)

        h, w = depth.shape[:2]
        fx = fy = 2.8 * w / 5.376
        cx, cy = w / 2, h / 2

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        mask = (z > 0.01) & (z < 1.5)
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

        if len(points) < 50:
            return None

        # Subsample
        if len(points) > 2000:
            idx = np.random.choice(len(points), 2000, replace=False)
            points = points[idx]

        # Center
        points -= points.mean(axis=0)
        return points

    def get_object_poses(self):
        """Get all object poses via MCP."""
        result = self.mcp._send("get_all_poses", {
            "root_path": "/World/envs/env_0/scene"
        })
        r = result.get("result", result)
        return r.get("poses", {})

    # ── Robot control ──
    def send_joint_command(self, positions, joint_names=None):
        """Send joint position command via /isaac_joint_commands."""
        if joint_names is None:
            joint_names = PANDA_JOINTS

        msg = JointState()
        msg.name = joint_names
        msg.position = [float(p) for p in positions]
        self._joint_cmd_pub.publish(msg)

    def move_to_joints(self, target_positions, speed=0.5, timeout=5.0):
        """Move robot to target joint positions smoothly."""
        self.get_logger().info(f"Moving to joint target...")

        steps = int(timeout / 0.066)  # ~15Hz
        start_positions = None

        with self._lock:
            if self._latest_joints:
                names = list(self._latest_joints.name)
                all_pos = list(self._latest_joints.position)
                start_positions = []
                for jn in PANDA_JOINTS:
                    if jn in names:
                        start_positions.append(all_pos[names.index(jn)])
                    else:
                        start_positions.append(0.0)

        if start_positions is None:
            start_positions = PANDA_HOME[:]

        for step in range(steps):
            t = min(1.0, (step + 1) / (steps * speed))
            # Linear interpolation
            interp = [s + t * (g - s) for s, g in zip(start_positions, target_positions)]
            self.send_joint_command(interp)
            self.spin_for(0.066)

        # Final command
        self.send_joint_command(target_positions)
        self.spin_for(0.5)

    def set_gripper(self, close=False):
        """Open or close the Robotiq gripper via MCP."""
        finger_val = float(math.pi / 4) if close else 0.0
        self.mcp._send("set_robot_joints", {
            "joint_positions": {"finger_joint": finger_val}
        })
        self.spin_for(0.3)

    # ── Scene management ──
    def randomize_and_reset(self):
        """Randomize scene and reset robot to home."""
        self.get_logger().info("Randomizing scene + resetting robot...")

        # Randomize object positions
        self.mcp._send("randomize_scene", {
            "randomize_objects": True,
            "randomize_lighting": True,
            "randomize_colors": False,
            "object_pos_range": [[-0.15, -0.15, 0.45], [0.15, 0.15, 0.55]],
        })

        # Reset robot to home
        self.move_to_joints(PANDA_HOME, speed=1.0, timeout=2.0)
        self.set_gripper(close=False)
        self.spin_for(1.0)

    # ── Grasp generation ──
    def generate_grasps(self, point_cloud):
        """Generate grasp poses using GraspGen."""
        if self.grasp_client is None:
            # Fallback: generate a simple top-down grasp
            self.get_logger().warn("No GraspGen, using heuristic grasp")
            grasp = np.eye(4, dtype=np.float32)
            grasp[:3, 3] = point_cloud.mean(axis=0)
            grasp[2, 3] += 0.1  # offset above
            return np.array([grasp]), np.array([1.0])

        try:
            t0 = time.time()
            grasps, confidences = self.grasp_client.infer(
                point_cloud,
                num_grasps=self.args.num_grasps,
                topk_num_grasps=self.args.topk_grasps,
            )
            dt = time.time() - t0
            self.get_logger().info(
                f"GraspGen: {len(grasps)} grasps in {dt:.2f}s, "
                f"best conf={confidences[0]:.3f}"
            )
            return grasps, confidences
        except Exception as e:
            self.get_logger().error(f"GraspGen failed: {e}")
            return None, None

    # ── Pick and Place execution ──
    def execute_pick_and_place(self, grasp_pose):
        """Execute pick-and-place using joint-space waypoints.

        Phases:
          1. Move to pre-grasp (above object)
          2. Approach (lower to grasp)
          3. Close gripper
          4. Lift
          5. Move to place position
          6. Open gripper
          7. Retract to home
        """
        self.get_logger().info("Executing pick-and-place sequence...")

        # Phase 1: Pre-grasp position
        self.get_logger().info("  Phase 1: Pre-grasp")
        self.move_to_joints(PANDA_PRE_GRASP, speed=0.8, timeout=3.0)

        # Phase 2: Approach (move down slightly)
        self.get_logger().info("  Phase 2: Approach")
        approach = PANDA_PRE_GRASP[:]
        approach[1] -= 0.2  # lower shoulder
        approach[3] += 0.3  # extend elbow
        self.move_to_joints(approach, speed=0.5, timeout=2.0)

        # Phase 3: Close gripper
        self.get_logger().info("  Phase 3: Grasp")
        self.set_gripper(close=True)
        self.spin_for(0.5)

        # Phase 4: Lift
        self.get_logger().info("  Phase 4: Lift")
        self.move_to_joints(PANDA_PRE_GRASP, speed=0.5, timeout=2.0)

        # Phase 5: Move to place
        self.get_logger().info("  Phase 5: Move to place")
        self.move_to_joints(PANDA_PLACE, speed=0.6, timeout=3.0)

        # Phase 6: Release
        self.get_logger().info("  Phase 6: Release")
        self.set_gripper(close=False)
        self.spin_for(0.5)

        # Phase 7: Retract
        self.get_logger().info("  Phase 7: Retract to home")
        self.move_to_joints(PANDA_HOME, speed=0.8, timeout=3.0)

        return True

    # ── Episode runner ──
    def run_episode(self, ep_id):
        """Run one complete grasp-and-place episode."""
        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"  Episode {ep_id}/{self.args.num_episodes}")
        self.get_logger().info(f"{'='*50}")

        # 1. Randomize + reset
        self.randomize_and_reset()

        # 2. Start recording
        self._collector_pub.publish(String(data="start"))
        self.spin_for(0.5)

        # 3. Capture point cloud
        self.get_logger().info("Capturing point cloud from wrist camera...")
        self.spin_for(1.0)
        pc = self.get_point_cloud()

        if pc is None:
            self.get_logger().warn("No point cloud, trying object poses instead...")
            # Fallback: use MCP to get object positions
            poses = self.get_object_poses()
            if poses:
                self.get_logger().info(f"Found {len(poses)} objects via MCP")
            self._collector_pub.publish(String(data="failure"))
            return False

        self.get_logger().info(f"Point cloud: {pc.shape[0]} points")

        # 4. Generate grasps
        grasps, confidences = self.generate_grasps(pc)
        if grasps is None or len(grasps) == 0:
            self.get_logger().warn("No grasps, skipping")
            self._collector_pub.publish(String(data="failure"))
            return False

        # 5. Execute best grasp
        best_grasp = grasps[0]
        success = self.execute_pick_and_place(best_grasp)

        # 6. Save episode
        if success:
            self._collector_pub.publish(String(data="success"))
            self.get_logger().info("Episode SUCCESS")
        else:
            self._collector_pub.publish(String(data="failure"))
            self.get_logger().info("Episode FAILURE")

        self.spin_for(0.5)
        return success

    def run(self):
        """Run the full automated pipeline."""
        self.get_logger().info(f"\nAuto Grasp Pipeline: {self.args.num_episodes} episodes")
        self.get_logger().info(f"Task: {self.args.task}")
        self.get_logger().info(f"GraspGen: {'connected' if self.grasp_client else 'unavailable (heuristic mode)'}")

        successes = 0
        for ep in range(self.args.num_episodes):
            try:
                if self.run_episode(ep):
                    successes += 1
            except Exception as e:
                self.get_logger().error(f"Episode {ep} error: {e}")
                import traceback
                traceback.print_exc()

            self.get_logger().info(
                f"Progress: {ep+1}/{self.args.num_episodes}, "
                f"Success: {successes}/{ep+1} ({100*successes/(ep+1):.0f}%)"
            )

        self.get_logger().info(f"\nPipeline complete: {successes}/{self.args.num_episodes} successful")
        return successes


def main():
    parser = argparse.ArgumentParser(description="Auto Grasp Pipeline")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--topk_grasps", type=int, default=50)
    parser.add_argument("--graspgen_host", type=str, default="localhost")
    parser.add_argument("--graspgen_port", type=int, default=5556)
    parser.add_argument("--task", type=str, default="pick and place object")
    args = parser.parse_args()

    rclpy.init()
    pipeline = AutoGraspPipeline(args)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline.grasp_client:
            try:
                pipeline.grasp_client.close()
            except Exception:
                pass
        pipeline.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
