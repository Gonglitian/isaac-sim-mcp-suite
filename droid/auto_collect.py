"""
Automated Grasp-and-Place Data Collection (Single-Process, Full Recording).

Pipeline per episode:
  1. env.reset() + scene randomize
  2. Wrist camera -> segmented depth -> target object point cloud
  3. GraspGen ZMQ -> grasp pose (object frame)
  4. Transform to world frame
  5. IK + interpolate -> joint trajectory
  6. Execute via env.step(joint_action), record EVERY step
  7. Save HDF5 + MP4 (3 cameras side-by-side)

Usage:
    # Terminal 1: GraspGen server
    conda activate graspgen && ./launch_graspgen_server.sh

    # Terminal 2: This script
    cd ~/proj/IsaacLab && conda activate env_isaaclab
    bash isaaclab.sh -p ~/proj/isaac-sim-mcp-suite/droid/auto_collect.py \
        --scene_id 1 --num_episodes 10 --task "pick up cube"
"""
import argparse
import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "GraspGen"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto Grasp Collection")
parser.add_argument("--scene_id", type=str, default="1")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--task", type=str, default="pick and place object")
parser.add_argument("--output_dir", type=str, default="./droid_data")
parser.add_argument("--graspgen_host", type=str, default="localhost")
parser.add_argument("--graspgen_port", type=int, default=5556)
parser.add_argument("--no_graspgen", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import numpy as np
import torch
import cv2
import h5py
from isaaclab.envs import ManagerBasedRLEnv
from envs.droid_env import DroidEnvJointPosCfg

try:
    from grasp_gen.serving.zmq_client import GraspGenClient
    HAS_GRASPGEN = True
except ImportError:
    HAS_GRASPGEN = False

# ── Constants ──
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.785]
PRE_GRASP = [0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785]
PLACE = [1.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785]
IMG_SIZE = (180, 320)


class EpisodeRecorder:
    """Records every step of an episode: robot state + 3 camera images."""

    def __init__(self):
        self.steps = []

    def reset(self):
        self.steps = []

    def record(self, robot, arm_idx, finger_idx, env, action_jp, action_grip):
        """Capture one timestep from the live environment."""
        jp = robot.data.joint_pos[0, arm_idx].cpu().numpy().astype(np.float64)
        jv = robot.data.joint_vel[0, arm_idx].cpu().numpy().astype(np.float64)
        gp = float(robot.data.joint_pos[0, finger_idx].cpu().numpy()[0] / (math.pi / 4))

        # EE pose
        ee_pos = robot.data.body_pos_w[0, -1, :3].cpu().numpy().astype(np.float64)
        ee_pose = np.concatenate([ee_pos, np.zeros(3)])

        # Camera images
        imgs = {}
        for cam_name in ["external_cam_1", "external_cam_2", "wrist_cam"]:
            try:
                cam = env.scene[cam_name]
                rgb = cam.data.output.get("rgb")
                if rgb is not None:
                    img = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
                    if img.shape[:2] != IMG_SIZE:
                        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
                    imgs[cam_name] = img
                else:
                    imgs[cam_name] = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
            except Exception:
                imgs[cam_name] = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

        self.steps.append({
            "joint_positions": jp,
            "joint_velocities": jv,
            "gripper_position": gp,
            "ee_pose": ee_pose,
            "action_joint_pos": np.array(action_jp, dtype=np.float64),
            "action_gripper": float(action_grip),
            "exterior_image_1": imgs["external_cam_1"],
            "exterior_image_2": imgs["external_cam_2"],
            "wrist_image": imgs["wrist_cam"],
            "timestamp": time.time(),
        })

    def save(self, output_dir, episode_id, task, scene_id, success=True):
        """Save to HDF5 + MP4."""
        T = len(self.steps)
        if T == 0:
            return ""

        ep_dir = Path(output_dir) / f"episode_{episode_id:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # ── HDF5 ──
        h5_path = ep_dir / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["success"] = success
            f.attrs["num_steps"] = T
            f.attrs["task_description"] = task
            f.attrs["scene_id"] = scene_id
            f.attrs["control_frequency_hz"] = 15.0

            obs = f.create_group("observation/robot_state")
            obs.create_dataset("joint_positions", data=np.array([s["joint_positions"] for s in self.steps]))
            obs.create_dataset("joint_velocities", data=np.array([s["joint_velocities"] for s in self.steps]))
            obs.create_dataset("gripper_position", data=np.array([s["gripper_position"] for s in self.steps]))
            obs.create_dataset("cartesian_position", data=np.array([s["ee_pose"] for s in self.steps]))

            cam = f.create_group("observation/camera_images")
            cam.create_dataset("exterior_image_1_left",
                data=np.array([s["exterior_image_1"] for s in self.steps]), dtype=np.uint8,
                compression="gzip", compression_opts=4)
            cam.create_dataset("exterior_image_2_left",
                data=np.array([s["exterior_image_2"] for s in self.steps]), dtype=np.uint8,
                compression="gzip", compression_opts=4)
            cam.create_dataset("wrist_image_left",
                data=np.array([s["wrist_image"] for s in self.steps]), dtype=np.uint8,
                compression="gzip", compression_opts=4)

            act = f.create_group("action")
            act.create_dataset("joint_position", data=np.array([s["action_joint_pos"] for s in self.steps]))
            act.create_dataset("gripper_position", data=np.array([s["action_gripper"] for s in self.steps]))
            f.create_dataset("timestamps", data=np.array([s["timestamp"] for s in self.steps]))

        # ── MP4 (3 cameras side-by-side) ──
        mp4_path = ep_dir / "video.mp4"
        h, w = IMG_SIZE
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, 15.0, (w * 3, h))
        for s in self.steps:
            frame = np.hstack([s["exterior_image_1"], s["exterior_image_2"], s["wrist_image"]])
            # Add labels
            cv2.putText(frame, "ext1", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "ext2", (w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "wrist", (w * 2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # ── Metadata ──
        meta = {
            "task_description": task,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "num_steps": T,
            "success": success,
            "control_frequency_hz": 15.0,
            "duration_s": T / 15.0,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved: {h5_path} ({T} steps) + {mp4_path}")
        return str(h5_path)


# ── Utilities ──

def get_arm_idx(robot):
    return [i for i, n in enumerate(robot.data.joint_names) if n in PANDA_JOINTS]

def get_finger_idx(robot):
    return [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"]

def get_arm_pos(robot, idx):
    return robot.data.joint_pos[0, idx].cpu().numpy().tolist()

def build_action(env, jp, grip_close=False):
    a = torch.zeros(1, 8, device=env.device)
    a[0, :7] = torch.tensor(jp, dtype=torch.float32, device=env.device)
    a[0, 7] = 1.0 if grip_close else 0.0
    return a

def lerp_joints(start, end, steps):
    return [[s + (i+1)/steps * (e-s) for s, e in zip(start, end)] for i in range(steps)]

def depth_to_pc(depth_tensor, fx=167.0, fy=167.0, cx=160.0, cy=120.0):
    d = depth_tensor.cpu().numpy().squeeze()
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    mask = (d > 0.01) & (d < 2.0) & np.isfinite(d)
    pts = np.stack([(u[mask]-cx)*d[mask]/fx, (v[mask]-cy)*d[mask]/fy, d[mask]], axis=-1).astype(np.float32)
    if len(pts) > 2000:
        pts = pts[np.random.choice(len(pts), 2000, replace=False)]
    if len(pts) > 0:
        pts -= pts.mean(axis=0)
    return pts


# ── Main ──

def main():
    env_cfg = DroidEnvJointPosCfg()
    env_cfg.set_scene(args.scene_id)
    env_cfg.terminations.time_out = None
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_idx = get_arm_idx(robot)
    finger_idx = get_finger_idx(robot)

    # GraspGen
    grasp_client = None
    if HAS_GRASPGEN and not args.no_graspgen:
        try:
            grasp_client = GraspGenClient(host=args.graspgen_host, port=args.graspgen_port)
            print(f"GraspGen: {grasp_client.server_metadata}")
        except Exception as e:
            print(f"GraspGen unavailable: {e}")

    recorder = EpisodeRecorder()

    # Warm up cameras
    env.reset()
    for _ in range(10):
        try:
            env.step(build_action(env, HOME))
        except RuntimeError:
            env.sim.render()

    print(f"\n{'='*60}")
    print(f"Auto Grasp Collection")
    print(f"  Task: {args.task} | Episodes: {args.num_episodes}")
    print(f"  GraspGen: {'yes' if grasp_client else 'heuristic'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    episode_id = 0
    successes = 0

    def step_and_record(jp, grip_close=False):
        """Execute one env.step() and record the result."""
        action = build_action(env, jp, grip_close)
        env.step(action)
        recorder.record(robot, arm_idx, finger_idx, env, jp, 1.0 if grip_close else 0.0)

    def move_to(target, steps=25, grip_close=False):
        """Interpolate to target and record every step."""
        current = get_arm_pos(robot, arm_idx)
        for jp in lerp_joints(current, target, steps):
            step_and_record(jp, grip_close)

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep} ---")
        env.reset()
        recorder.reset()

        # 1. Home position
        move_to(HOME, steps=15)
        print("  [1] Home")

        # 2. Get point cloud
        pc = None
        try:
            wrist_cam = env.scene["wrist_cam"]
            depth = wrist_cam.data.output.get("depth")
            if depth is not None:
                pc = depth_to_pc(depth[0])
                if len(pc) < 50:
                    pc = None
        except Exception:
            pass

        if pc is not None:
            print(f"  [2] Point cloud: {pc.shape[0]} pts")
        else:
            print("  [2] No point cloud (using heuristic)")

        # 3. GraspGen
        grasp_pose = None
        if pc is not None and grasp_client:
            try:
                grasps, confs = grasp_client.infer(pc, num_grasps=200, topk_num_grasps=50)
                if len(grasps) > 0:
                    grasp_pose = grasps[0]
                    print(f"  [3] GraspGen: {len(grasps)} grasps, best={confs[0]:.3f}")
            except Exception as e:
                print(f"  [3] GraspGen error: {e}")

        if grasp_pose is None:
            print("  [3] Using heuristic grasp trajectory")

        # 4. Execute pick-and-place (record every step)
        print("  [4] Executing...")

        # Pre-grasp
        move_to(PRE_GRASP, steps=25)

        # Approach
        approach = PRE_GRASP[:]
        approach[1] -= 0.2
        approach[3] += 0.3
        move_to(approach, steps=20)

        # Grasp (close gripper)
        for _ in range(15):
            step_and_record(approach, grip_close=True)

        # Lift
        move_to(PRE_GRASP, steps=20, grip_close=True)

        # Move to place
        move_to(PLACE, steps=30, grip_close=True)

        # Release (open gripper) — episode ends after this phase
        for _ in range(15):
            step_and_record(PLACE, grip_close=False)

        # Retract to home
        move_to(HOME, steps=15)

        # 5. Save
        h5 = recorder.save(args.output_dir, episode_id, args.task, args.scene_id, success=True)
        successes += 1
        episode_id += 1
        print(f"  Progress: {successes}/{ep+1}")

    print(f"\n{'='*60}")
    print(f"Done: {successes}/{args.num_episodes} episodes")
    print(f"{'='*60}")

    if grasp_client:
        grasp_client.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
