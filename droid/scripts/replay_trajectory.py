"""
Replay a recorded DROID trajectory in IsaacLab.

Loads trajectory.h5, sets robot joint positions each step, and optionally
shows camera images in a side-by-side viewer.

Usage:
    cd ~/proj/IsaacLab
    conda activate env_isaaclab
    bash isaaclab.sh -p ~/proj/isaacsim-ros2-moveit/droid_sim/scripts/replay_trajectory.py \
        --trajectory ~/proj/isaacsim-ros2-moveit/droid_data/episode_0000/trajectory.h5 \
        --scene_id 1 --speed 1.0
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay DROID Trajectory")
parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory.h5")
parser.add_argument("--scene_id", type=str, default="1")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--show_images", action="store_true", help="Show camera images in OpenCV window")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from envs.droid_env import DroidEnvCfg


def main():
    # Load trajectory
    print(f"\nLoading trajectory: {args.trajectory}")
    with h5py.File(args.trajectory, "r") as f:
        joint_positions = f["observation/robot_state/joint_positions"][:]
        gripper_positions = f["observation/robot_state/gripper_position"][:]
        T = joint_positions.shape[0]
        hz = f.attrs.get("control_frequency_hz", 15.0)
        task = f.attrs.get("task_description", "unknown")
        success = f.attrs.get("success", False)

        # Load images if requested
        images = None
        if args.show_images and "observation/camera_images/exterior_image_1_left" in f:
            images = {
                "ext1": f["observation/camera_images/exterior_image_1_left"][:],
                "ext2": f["observation/camera_images/exterior_image_2_left"][:],
                "wrist": f["observation/camera_images/wrist_image_left"][:],
            }

        # Load object poses if available
        obj_poses = {}
        if "observation/object_poses" in f:
            for obj_name in f["observation/object_poses"]:
                obj_poses[obj_name] = {
                    "position": f[f"observation/object_poses/{obj_name}/position"][:],
                    "orientation": f[f"observation/object_poses/{obj_name}/orientation"][:],
                }

    print(f"  Task: {task}")
    print(f"  Steps: {T}, Duration: {T/hz:.1f}s, Hz: {hz}")
    print(f"  Success: {success}")
    print(f"  Objects tracked: {list(obj_poses.keys())}")

    # Create environment
    env_cfg = DroidEnvCfg()
    env_cfg.set_scene(args.scene_id)
    env_cfg.terminations.time_out = None
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    # Get robot and joint info
    robot = env.scene["robot"]
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    arm_idx = [i for i, n in enumerate(robot.data.joint_names) if n in arm_names]
    finger_idx = [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"]

    print(f"\n=== Replaying ({args.speed}x speed) ===")
    print("Press Ctrl+C to stop.\n")

    step_dt = 1.0 / (hz * args.speed)

    # OpenCV viewer
    cv2 = None
    if args.show_images and images is not None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("Warning: opencv not available, skipping image display")

    try:
        for step in range(T):
            t_start = time.time()

            # Set joint positions directly
            target_joint_pos = torch.zeros(1, len(robot.data.joint_names), device=env.device)
            for i, idx in enumerate(arm_idx):
                target_joint_pos[0, idx] = joint_positions[step, i]
            for idx in finger_idx:
                target_joint_pos[0, idx] = gripper_positions[step] * (np.pi / 4)

            robot.write_joint_state_to_sim(target_joint_pos, torch.zeros_like(target_joint_pos))

            # Step sim
            env.sim.step(render=True)
            env.scene.update(dt=env.step_dt)

            # Show images
            if cv2 is not None and images is not None:
                imgs = [images["ext1"][step], images["ext2"][step], images["wrist"][step]]
                # Add labels
                for img, label in zip(imgs, ["ext1", "ext2", "wrist"]):
                    cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                concat = np.hstack(imgs)
                concat = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Replay: {task}", concat)
                if cv2.waitKey(1) == ord("q"):
                    break

            # Progress
            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{T} ({(step+1)/hz:.1f}s)")

            # Timing
            elapsed = time.time() - t_start
            sleep_time = step_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if cv2 is not None:
        cv2.destroyAllWindows()

    print(f"\nReplay complete. {T} steps played.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
