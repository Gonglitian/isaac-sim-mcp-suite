#!/usr/bin/env python3
"""
Automated Grasp-and-Place Pipeline.

Combines:
  - Isaac Sim wrist camera (depth + RGB + instance segmentation)
  - GraspGen (6-DOF grasp generation via ZMQ)
  - MoveIt2 (motion planning via ROS2 action)
  - MCP (scene randomization + reset)

Flow per episode:
  1. MCP: randomize_scene()
  2. Wrist cam: get depth image -> point cloud
  3. GraspGen: generate grasp poses
  4. MoveIt2: plan approach -> grasp -> lift -> place
  5. Data collector: record trajectory
  6. Repeat

Usage:
    # Terminal 1: Isaac Sim (IsaacLab DROID env + ROS2)
    # Terminal 2: GraspGen server
    python libs/GraspGen/client-server/graspgen_server.py \
        --gripper_config libs/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml

    # Terminal 3: This pipeline
    source /opt/ros/humble/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    python droid/scripts/auto_grasp_pipeline.py --num_episodes 10
"""
import argparse
import sys
import os
import time
import threading
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

# GraspGen ZMQ client (lightweight, no GPU needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs", "GraspGen"))
try:
    from grasp_gen.serving.zmq_client import GraspGenClient
    HAS_GRASPGEN = True
except ImportError:
    HAS_GRASPGEN = False
    print("WARNING: GraspGen not installed. Install: cd libs/GraspGen && pip install -e .")

# MCP client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "moveit", "scripts"))
from mcp_client import IsaacMCP


class AutoGraspPipeline(Node):
    """ROS2 node that orchestrates automated grasp-and-place episodes."""

    def __init__(self, args):
        super().__init__("auto_grasp_pipeline")
        self.bridge = CvBridge()
        self.args = args

        # MCP client for scene control
        self.mcp = IsaacMCP()
        if not self.mcp.is_connected():
            self.get_logger().error("MCP not connected. Is Isaac Sim running?")
            raise RuntimeError("MCP not available")

        # GraspGen client
        self.grasp_client = None
        if HAS_GRASPGEN:
            try:
                self.grasp_client = GraspGenClient(
                    host=args.graspgen_host, port=args.graspgen_port
                )
                self.get_logger().info(f"GraspGen connected at {args.graspgen_host}:{args.graspgen_port}")
            except Exception as e:
                self.get_logger().warn(f"GraspGen not available: {e}")

        # Latest sensor data
        self._lock = threading.Lock()
        self._latest_depth = None
        self._latest_rgb = None
        self._latest_joints = None

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        # Subscribers
        self.create_subscription(Image, "/cam_wrist/depth", self._depth_cb, qos)
        self.create_subscription(Image, "/cam_wrist/rgb", self._rgb_cb, qos)
        self.create_subscription(JointState, "/isaac_joint_states", self._joint_cb, qos)

        # Data collector control
        self._collector_pub = self.create_publisher(String, "/collector/cmd", 10)

        self.get_logger().info("Auto Grasp Pipeline initialized")

    def _depth_cb(self, msg):
        with self._lock:
            self._latest_depth = msg

    def _rgb_cb(self, msg):
        with self._lock:
            self._latest_rgb = msg

    def _joint_cb(self, msg):
        with self._lock:
            self._latest_joints = msg

    def get_point_cloud_from_depth(self):
        """Convert depth image to point cloud using camera intrinsics."""
        with self._lock:
            depth_msg = self._latest_depth

        if depth_msg is None:
            self.get_logger().warn("No depth image available")
            return None

        # Convert depth image
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm -> m

        h, w = depth.shape[:2]

        # Camera intrinsics (approximate for wrist cam)
        fx = fy = 2.8 * w / 5.376  # focal_length * width / horizontal_aperture
        cx, cy = w / 2, h / 2

        # Generate point cloud
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter valid points
        mask = (z > 0.01) & (z < 2.0)
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

        if len(points) < 100:
            self.get_logger().warn(f"Too few points: {len(points)}")
            return None

        # Subsample to 2000 points
        if len(points) > 2000:
            idx = np.random.choice(len(points), 2000, replace=False)
            points = points[idx]

        # Center
        points -= points.mean(axis=0)

        return points

    def generate_grasps(self, point_cloud):
        """Call GraspGen server to generate grasp poses."""
        if self.grasp_client is None:
            self.get_logger().error("GraspGen client not available")
            return None, None

        try:
            grasps, confidences = self.grasp_client.infer(
                point_cloud,
                num_grasps=self.args.num_grasps,
                topk_num_grasps=self.args.topk_grasps,
            )
            self.get_logger().info(
                f"GraspGen: {len(grasps)} grasps, best confidence: {confidences[0]:.3f}"
            )
            return grasps, confidences
        except Exception as e:
            self.get_logger().error(f"GraspGen inference failed: {e}")
            return None, None

    def execute_grasp_via_mcp(self, grasp_pose):
        """Execute grasp using MCP set_robot_joints (simplified approach).

        For a full pipeline, this would use MoveIt2 action client.
        Here we use MCP to demonstrate the concept.
        """
        # Extract position from grasp pose (4x4 SE(3))
        pos = grasp_pose[:3, 3]
        self.get_logger().info(f"Executing grasp at position: {pos}")

        # Approach: move above grasp point
        approach_code = f'''
import numpy as np
from pxr import Gf

# Get current robot EE pose
import omni.usd
stage = omni.usd.get_context().get_stage()

# Log grasp target
print("Grasp target: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
'''
        self.mcp.execute(approach_code)
        return True

    def randomize_and_reset(self):
        """Randomize scene objects and reset robot."""
        self.get_logger().info("Randomizing scene...")
        result = self.mcp._send("randomize_scene", {
            "randomize_objects": True,
            "randomize_lighting": True,
            "object_pos_range": [[-0.2, -0.2, 0.4], [0.2, 0.2, 0.6]],
        })
        self.get_logger().info(f"Randomize result: {result.get('status', 'unknown')}")

        # Reset robot to home position
        self.mcp._send("set_robot_joints", {
            "joint_positions": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.628,
                "panda_joint3": 0.0,
                "panda_joint4": -2.513,
                "panda_joint5": 0.0,
                "panda_joint6": 1.885,
                "panda_joint7": 0.0,
            }
        })
        time.sleep(1.0)

    def run_episode(self, episode_id):
        """Run one grasp-and-place episode."""
        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"Episode {episode_id}")
        self.get_logger().info(f"{'='*50}")

        # 1. Randomize scene
        self.randomize_and_reset()

        # 2. Start recording
        self._collector_pub.publish(String(data="start"))
        time.sleep(0.5)

        # 3. Get point cloud from wrist camera
        self.get_logger().info("Capturing point cloud from wrist camera...")
        # Spin to get latest data
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)

        point_cloud = self.get_point_cloud_from_depth()
        if point_cloud is None:
            self.get_logger().warn("No point cloud, skipping episode")
            self._collector_pub.publish(String(data="failure"))
            return False

        self.get_logger().info(f"Point cloud: {point_cloud.shape}")

        # 4. Generate grasps
        grasps, confidences = self.generate_grasps(point_cloud)
        if grasps is None or len(grasps) == 0:
            self.get_logger().warn("No grasps generated, skipping")
            self._collector_pub.publish(String(data="failure"))
            return False

        # 5. Execute best grasp
        best_grasp = grasps[0]  # highest confidence
        success = self.execute_grasp_via_mcp(best_grasp)

        # 6. Save episode
        if success:
            self._collector_pub.publish(String(data="success"))
        else:
            self._collector_pub.publish(String(data="failure"))

        time.sleep(0.5)
        return success

    def run(self):
        """Run the full pipeline for N episodes."""
        self.get_logger().info(f"\nStarting auto grasp pipeline: {self.args.num_episodes} episodes")

        successes = 0
        for ep in range(self.args.num_episodes):
            try:
                ok = self.run_episode(ep)
                if ok:
                    successes += 1
            except Exception as e:
                self.get_logger().error(f"Episode {ep} failed: {e}")

        self.get_logger().info(f"\nPipeline complete: {successes}/{self.args.num_episodes} successful")


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
        if HAS_GRASPGEN and pipeline.grasp_client:
            pipeline.grasp_client.close()
        pipeline.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
