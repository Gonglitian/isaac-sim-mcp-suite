"""DROID-style data collection with keyboard teleoperation.

Usage (from IsaacLab root):
    cd ~/proj/IsaacLab
    conda activate env_isaaclab
    bash isaaclab.sh -p ~/proj/isaacsim-ros2-moveit/droid_sim/collect_data.py \
        --scene_id 1 --task "put the cube in the bowl" --num_episodes 5

Keyboard controls (must focus the Isaac Sim viewport window):
    W/S = move X, A/D = move Y, Q/E = move Z
    Z/X = rotate roll, T/G = rotate pitch, C/V = rotate yaw
    K   = toggle gripper
    R   = reset episode
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DROID Data Collection")
parser.add_argument("--scene_id", type=str, default="1", choices=["1", "2", "3"])
parser.add_argument("--task", type=str, default="manipulation task")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="./droid_data")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Imports after sim init ---
import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv

from envs.droid_env import DroidEnvCfg
from recorder.droid_recorder import DroidRecorder


def main():
    # --- Environment ---
    env_cfg = DroidEnvCfg()
    env_cfg.set_scene(args.scene_id)
    env_cfg.terminations.time_out = None  # no auto-timeout during teleop
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- Keyboard teleop ---
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05))

    # --- State flags ---
    should_reset = False
    episode_done = False
    episode_success = False

    def on_reset():
        nonlocal should_reset
        should_reset = True
        print("[Teleop] Reset triggered")

    def on_success():
        nonlocal episode_done, episode_success
        episode_done = True
        episode_success = True
        print("[Teleop] Episode marked SUCCESS")

    def on_failure():
        nonlocal episode_done, episode_success
        episode_done = True
        episode_success = False
        print("[Teleop] Episode marked FAILURE")

    teleop.add_callback("R", on_reset)
    teleop.add_callback("SPACE", on_success)
    teleop.add_callback("ESCAPE", on_failure)

    # --- Recorder ---
    recorder = DroidRecorder(output_dir=args.output_dir)

    # --- Init ---
    obs, _ = env.reset()
    teleop.reset()
    episode_count = 0
    recorder.start_episode(task_description=args.task, scene_id=args.scene_id)

    print("\n" + "=" * 60)
    print("DROID Data Collection - Keyboard Teleoperation")
    print("=" * 60)
    print(f"Scene: {args.scene_id}  |  Task: '{args.task}'")
    print(f"Output: {args.output_dir}")
    print("-" * 60)
    print("W/S=X  A/D=Y  Q/E=Z  |  Z/X=roll  T/G=pitch  C/V=yaw")
    print("K=gripper  R=reset  SPACE=success  ESC=fail")
    print("=" * 60 + "\n")

    # --- Main loop (follows official IsaacLab teleop pattern) ---
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get teleop command: 7D tensor [dx,dy,dz,rx,ry,rz,gripper]
            action = teleop.advance()
            actions = action.unsqueeze(0).repeat(env.num_envs, 1)

            # Step env
            obs, _, terminated, truncated, info = env.step(actions)

            # Record timestep
            robot = env.scene["robot"]
            arm_names = [f"panda_joint{i}" for i in range(1, 8)]
            arm_idx = [i for i, n in enumerate(robot.data.joint_names) if n in arm_names]
            finger_idx = [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"]

            joint_pos = robot.data.joint_pos[0, arm_idx].cpu().numpy()
            joint_vel = robot.data.joint_vel[0, arm_idx].cpu().numpy()
            gripper_val = robot.data.joint_pos[0, finger_idx].cpu().numpy()[0] / (np.pi / 4)

            # EE pose from body state
            ee_pos = robot.data.body_pos_w[0, -1, :3].cpu().numpy()
            ee_pose = np.concatenate([ee_pos, np.zeros(3)])  # rpy placeholder

            action_np = action.cpu().numpy()
            action_gripper = 0.0 if action_np[6] > 0 else 1.0

            # Camera images
            cam_keys = {"external_cam_1", "external_cam_2", "wrist_cam"}
            imgs = {}
            policy_obs = obs.get("policy", {})
            for cam in cam_keys:
                if cam in policy_obs:
                    img = policy_obs[cam][0].cpu().numpy()
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    imgs[cam] = img
                else:
                    imgs[cam] = np.zeros((180, 320, 3), dtype=np.uint8)

            recorder.record_timestep(
                joint_positions=joint_pos,
                joint_velocities=joint_vel,
                gripper_position=gripper_val,
                ee_pose=ee_pose,
                action_joint_pos=joint_pos,
                action_gripper=action_gripper,
                exterior_image_1=imgs["external_cam_1"],
                exterior_image_2=imgs["external_cam_2"],
                wrist_image=imgs["wrist_cam"],
            )

            # Handle reset
            if should_reset:
                obs, _ = env.reset()
                teleop.reset()
                should_reset = False
                recorder.end_episode(success=False)
                episode_count += 1
                if episode_count >= args.num_episodes:
                    break
                recorder.start_episode(task_description=args.task, scene_id=args.scene_id)

            # Handle episode done (success/failure)
            if episode_done:
                recorder.end_episode(success=episode_success)
                episode_count += 1
                episode_done = False
                if episode_count >= args.num_episodes:
                    break
                obs, _ = env.reset()
                teleop.reset()
                recorder.start_episode(task_description=args.task, scene_id=args.scene_id)

    # Cleanup
    if recorder._recording:
        recorder.end_episode(success=False)

    env.close()
    print(f"\nDone. {episode_count} episodes saved to {args.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
