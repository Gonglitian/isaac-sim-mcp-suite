#!/usr/bin/env python3
"""
DROID 3-camera viewer: horizontally stacked ext1 | ext2 | wrist.

Usage:
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    python3 droid_sim/scripts/camera_viewer.py
"""
import sys
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraViewer(Node):
    def __init__(self):
        super().__init__("droid_camera_viewer")
        self.bridge = CvBridge()
        self._lock = threading.Lock()

        # Latest frames
        self._frames = {
            "ext1": None,
            "ext2": None,
            "wrist": None,
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, "/cam_ext1/rgb", lambda m: self._cb("ext1", m), qos)
        self.create_subscription(Image, "/cam_ext2/rgb", lambda m: self._cb("ext2", m), qos)
        self.create_subscription(Image, "/cam_wrist/rgb", lambda m: self._cb("wrist", m), qos)
        self.get_logger().info("Subscribed to 3 cameras. Waiting for images...")

    def _cb(self, name, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._frames[name] = img
        except Exception as e:
            self.get_logger().warn(f"{name}: {e}")

    def get_concat_frame(self, target_h=240):
        """Get horizontally concatenated frame: ext1 | ext2 | wrist."""
        with self._lock:
            frames = dict(self._frames)

        imgs = []
        for name in ["ext1", "ext2", "wrist"]:
            f = frames[name]
            if f is None:
                f = np.zeros((target_h, 320, 3), dtype=np.uint8)
                cv2.putText(f, f"No {name}", (80, target_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            else:
                h, w = f.shape[:2]
                scale = target_h / h
                f = cv2.resize(f, (int(w * scale), target_h))
            # Label
            cv2.putText(f, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            imgs.append(f)

        return np.hstack(imgs)


def main():
    rclpy.init()
    viewer = CameraViewer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(viewer,), daemon=True)
    spin_thread.start()

    print("\n=== DROID 3-Camera Viewer ===")
    print("Press 'q' in the window to quit.\n")

    while rclpy.ok():
        frame = viewer.get_concat_frame()
        cv2.imshow("DROID Cameras: ext1 | ext2 | wrist", frame)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    viewer.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
