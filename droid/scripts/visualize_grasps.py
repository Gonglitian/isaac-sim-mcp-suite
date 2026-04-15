#!/usr/bin/env python3
"""
Visualize saved point cloud + grasp poses in 3D using viser.

Usage:
    conda activate graspgen
    python droid/scripts/visualize_grasps.py \
        --episode_dir droid_data/episode_0000 \
        --port 8080

Then open http://localhost:8080 in your browser.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize grasps with viser")
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gripper", type=str, default="robotiq_2f_140")
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir)

    # Load saved data
    pc = np.load(ep_dir / "point_cloud.npy")
    grasps = np.load(ep_dir / "grasps.npy")
    confs = np.load(ep_dir / "confidences.npy")

    print(f"Point cloud: {pc.shape}")
    print(f"Grasps: {grasps.shape}, confidences: {confs.shape}")
    print(f"Best confidence: {confs[0]:.3f}")

    # Create viser visualizer
    vis = create_visualizer(port=args.port)

    # Visualize point cloud
    pc_color = np.ones((len(pc), 3), dtype=np.uint8) * 180
    visualize_pointcloud(vis, "point_cloud", pc, pc_color, size=0.003)

    # Visualize grasps (colored by confidence)
    scores = get_color_from_score(confs, use_255_scale=True)
    for i, grasp in enumerate(grasps):
        g = grasp.copy()
        g[3, 3] = 1.0
        visualize_grasp(
            vis, f"grasps/{i:03d}", g,
            color=scores[i], gripper_name=args.gripper, linewidth=0.6,
        )

    print(f"\nViser running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
