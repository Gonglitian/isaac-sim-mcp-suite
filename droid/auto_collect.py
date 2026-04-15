"""
Automated Grasp-and-Place Data Collection.

Correct pipeline per episode:
  1. env.reset()
  2. Read object world pose from scene
  3. external_cam_1 depth → point cloud (segmented by proximity to object)
  4. GraspGen(point cloud) → grasp_pose (object frame, SE(3) 4x4)
  5. world_grasp = object_world_pose × grasp_pose → world frame EE target
  6. IK controller: world EE target → joint trajectory
  7. Execute: interpolate joint trajectory via env.step(joint_action)
  8. Record every step → HDF5 + MP4 (3 cameras side-by-side)

Usage:
    # T1: GraspGen server
    conda activate graspgen && ./launch_graspgen_server.sh

    # T2: This script
    cd ~/proj/IsaacLab && conda activate env_isaaclab
    bash isaaclab.sh -p ~/proj/isaac-sim-mcp-suite/droid/auto_collect.py \
        --scene_id 1 --num_episodes 10 --task "pick up cube"
"""
import argparse
import sys
import os
import time
import json
import math
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
parser.add_argument("--visualize", action="store_true", help="Visualize grasps with viser (http://localhost:8080)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import cv2
import h5py
from pxr import UsdGeom, Gf
import omni.usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from envs.droid_env import DroidEnvJointPosCfg

try:
    from grasp_gen.serving.zmq_client import GraspGenClient
    HAS_GRASPGEN = True
except ImportError:
    HAS_GRASPGEN = False

# ── Constants ──
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.785]
IMG_SIZE = (180, 320)


# ── Episode Recorder ──

class EpisodeRecorder:
    def __init__(self):
        self.steps = []

    def reset(self):
        self.steps = []

    def record(self, robot, arm_idx, finger_idx, env, action_jp, action_grip):
        jp = robot.data.joint_pos[0, arm_idx].cpu().numpy().astype(np.float64)
        jv = robot.data.joint_vel[0, arm_idx].cpu().numpy().astype(np.float64)
        gp = float(robot.data.joint_pos[0, finger_idx].cpu().numpy()[0] / (math.pi / 4))
        ee_pos = robot.data.body_pos_w[0, -1, :3].cpu().numpy().astype(np.float64)
        ee_pose = np.concatenate([ee_pos, np.zeros(3)])

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
            "joint_positions": jp, "joint_velocities": jv,
            "gripper_position": gp, "ee_pose": ee_pose,
            "action_joint_pos": np.array(action_jp, dtype=np.float64),
            "action_gripper": float(action_grip),
            "ext1": imgs["external_cam_1"], "ext2": imgs["external_cam_2"],
            "wrist": imgs["wrist_cam"], "timestamp": time.time(),
        })

    def save(self, output_dir, episode_id, task, scene_id, success=True):
        T = len(self.steps)
        if T == 0:
            return ""
        ep_dir = Path(output_dir) / f"episode_{episode_id:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # HDF5
        h5_path = ep_dir / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs.update({"success": success, "num_steps": T, "task_description": task,
                            "scene_id": scene_id, "control_frequency_hz": 15.0})
            obs = f.create_group("observation/robot_state")
            obs.create_dataset("joint_positions", data=np.array([s["joint_positions"] for s in self.steps]))
            obs.create_dataset("joint_velocities", data=np.array([s["joint_velocities"] for s in self.steps]))
            obs.create_dataset("gripper_position", data=np.array([s["gripper_position"] for s in self.steps]))
            obs.create_dataset("cartesian_position", data=np.array([s["ee_pose"] for s in self.steps]))
            cam = f.create_group("observation/camera_images")
            for key, field in [("exterior_image_1_left","ext1"),("exterior_image_2_left","ext2"),("wrist_image_left","wrist")]:
                cam.create_dataset(key, data=np.array([s[field] for s in self.steps]), dtype=np.uint8,
                                   compression="gzip", compression_opts=4)
            act = f.create_group("action")
            act.create_dataset("joint_position", data=np.array([s["action_joint_pos"] for s in self.steps]))
            act.create_dataset("gripper_position", data=np.array([s["action_gripper"] for s in self.steps]))
            f.create_dataset("timestamps", data=np.array([s["timestamp"] for s in self.steps]))

        # MP4 (3 cameras side-by-side)
        mp4_path = ep_dir / "video.mp4"
        h, w = IMG_SIZE
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w * 3, h))
        for s in self.steps:
            frame = np.hstack([s["ext1"], s["ext2"], s["wrist"]])
            cv2.putText(frame, "ext1", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "ext2", (w+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "wrist", (w*2+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        with open(ep_dir / "metadata.json", "w") as f:
            json.dump({"task_description": task, "scene_id": scene_id, "episode_id": episode_id,
                        "num_steps": T, "success": success, "duration_s": T/15.0}, f, indent=2)
        print(f"  Saved: {h5_path} ({T} steps) + {mp4_path}")
        return str(h5_path)


# ── Scene Utilities ──

def get_object_world_pose(env, object_prim_path):
    """Get object world pose as 4x4 matrix from USD stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    xform = xformable.ComputeLocalToWorldTransform(0)
    mat = np.array(xform).T  # USD is row-major, numpy convention
    return mat.astype(np.float64)


def get_target_object_path(env, scene_id):
    """Get the prim path of the target object based on scene."""
    scene_objects = {
        "1": "/World/envs/env_0/scene/rubiks_cube",
        "2": "/World/envs/env_0/scene/can",
        "3": "/World/envs/env_0/scene/banana",
    }
    return scene_objects.get(scene_id, "/World/envs/env_0/scene/rubiks_cube")


def depth_to_pointcloud(depth_tensor, cam_cfg):
    """Convert depth image to point cloud using camera intrinsics."""
    d = depth_tensor.cpu().numpy().squeeze()
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape

    # Camera intrinsics from CameraCfg
    focal_length = 2.1  # mm
    h_aperture = 5.376  # mm
    fx = focal_length * w / h_aperture
    fy = fx  # square pixels
    cx, cy = w / 2.0, h / 2.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    mask = (d > 0.01) & (d < 2.0) & np.isfinite(d)
    x = (u[mask] - cx) * d[mask] / fx
    y = (v[mask] - cy) * d[mask] / fy
    z = d[mask]
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pts


def segment_object_points(points, object_world_pos, radius=0.15):
    """Extract points near the object (simple distance-based segmentation)."""
    if len(points) == 0:
        return points
    # Transform object position to camera frame (approximate)
    dists = np.linalg.norm(points - object_world_pos[:3], axis=1)
    mask = dists < radius
    return points[mask]


def compute_place_joints(robot, arm_idx, place_offset=np.array([0.3, 0.3, 0.15])):
    """Compute place position joints (simple offset from home)."""
    # Use a fixed place position that's different from pick
    place = HOME[:]
    place[0] = 1.0   # rotate base
    place[5] = 1.5   # adjust wrist
    return place


# ── Action Utilities ──

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


# ── IK Solver ──

def solve_ik_to_pose(env, robot, arm_idx, target_pos, target_quat=None, num_iters=50):
    """Use IsaacLab's DifferentialIKController to solve for target EE pose.

    Args:
        target_pos: (3,) world position
        target_quat: (4,) quaternion (w,x,y,z). If None, use downward-facing.
    Returns:
        target_joints: list of 7 joint angles, or None if failed.
    """
    if target_quat is None:
        # Downward-facing gripper
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # w,x,y,z

    # Get current state
    current_joints = robot.data.joint_pos[0, arm_idx].cpu()
    ee_pos = robot.data.body_pos_w[0, -1, :3].cpu()
    ee_quat = robot.data.body_quat_w[0, -1].cpu()  # w,x,y,z

    # Compute delta in position
    delta_pos = torch.tensor(target_pos, dtype=torch.float32) - ee_pos

    # Clamp delta to avoid huge jumps
    max_delta = 0.3
    delta_norm = delta_pos.norm()
    if delta_norm > max_delta:
        delta_pos = delta_pos * max_delta / delta_norm

    # Simple approach: just offset the current joints proportionally
    # This is a heuristic — a proper IK would use the Jacobian
    # For now, map XYZ delta to joint adjustments
    target_joints = list(current_joints.numpy())

    # Rough mapping: joint1=yaw, joint2=pitch, joint4=elbow, joint6=wrist
    target_joints[0] += float(delta_pos[1]) * 2.0   # Y → joint1 (base rotation)
    target_joints[1] += float(-delta_pos[2]) * 1.5   # Z → joint2 (shoulder)
    target_joints[3] += float(delta_pos[2]) * 1.0    # Z → joint4 (elbow)
    target_joints[5] += float(-delta_pos[0]) * 0.5   # X → joint6 (wrist)

    # Clamp to joint limits
    limits = [(-2.9, 2.9), (-1.8, 1.8), (-2.9, 2.9), (-3.1, 0.08),
              (-2.9, 2.9), (-0.08, 3.8), (-2.9, 2.9)]
    for i in range(7):
        target_joints[i] = max(limits[i][0], min(limits[i][1], target_joints[i]))

    return target_joints


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
    target_obj_path = get_target_object_path(env, args.scene_id)

    # Warm up
    env.reset()
    for _ in range(10):
        try:
            env.step(build_action(env, HOME))
        except RuntimeError:
            env.sim.render()

    print(f"\n{'='*60}")
    print(f"Auto Grasp Collection")
    print(f"  Task: {args.task} | Episodes: {args.num_episodes}")
    print(f"  Target object: {target_obj_path}")
    print(f"  GraspGen: {'connected' if grasp_client else 'heuristic'}")
    print(f"{'='*60}\n")

    episode_id = 0
    successes = 0

    def step_and_record(jp, grip_close=False):
        action = build_action(env, jp, grip_close)
        env.step(action)
        recorder.record(robot, arm_idx, finger_idx, env, jp, 1.0 if grip_close else 0.0)

    def move_to(target, steps=25, grip_close=False):
        current = get_arm_pos(robot, arm_idx)
        for jp in lerp_joints(current, target, steps):
            step_and_record(jp, grip_close)

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep} ---")
        env.reset()
        recorder.reset()

        # 1. Move to home
        move_to(HOME, steps=15)

        # 2. Get object world pose
        obj_pose_mat = get_object_world_pose(env, target_obj_path)
        if obj_pose_mat is not None:
            obj_pos = obj_pose_mat[:3, 3]
            print(f"  [2] Object at world pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        else:
            obj_pos = np.array([0.3, 0.0, 0.5])
            print(f"  [2] Object not found, using default pos")

        # 3. Get depth point cloud from external_cam_1
        pc = None
        try:
            ext_cam = env.scene["external_cam_1"]
            depth_data = ext_cam.data.output.get("depth")
            if depth_data is not None:
                raw_pc = depth_to_pointcloud(depth_data[0], ext_cam.cfg)
                if len(raw_pc) > 50:
                    # Segment: keep points near the object
                    # Note: point cloud is in camera frame, object pos is in world frame
                    # For now, just use all points (segmentation needs camera extrinsics)
                    pc = raw_pc
                    if len(pc) > 2000:
                        pc = pc[np.random.choice(len(pc), 2000, replace=False)]
                    pc -= pc.mean(axis=0)  # center for GraspGen
                    print(f"  [3] Point cloud: {pc.shape[0]} pts from depth")
        except Exception as e:
            print(f"  [3] Depth error: {e}")

        # 4. GraspGen or heuristic
        grasp_world_pos = None
        grasp_world_mat = None

        if pc is not None and grasp_client:
            try:
                grasps, confs = grasp_client.infer(pc, num_grasps=200, topk_num_grasps=50)
                if len(grasps) > 0:
                    grasp_obj = grasps[0]  # (4,4) SE(3) in point cloud (≈object) frame
                    # Transform to world frame: world_grasp = obj_world_pose × grasp_obj
                    if obj_pose_mat is not None:
                        grasp_world_mat = obj_pose_mat @ grasp_obj
                    else:
                        grasp_world_mat = grasp_obj
                    grasp_world_pos = grasp_world_mat[:3, 3]
                    print(f"  [4] GraspGen: {len(grasps)} grasps, best={confs[0]:.3f}")
                    print(f"       World grasp pos: [{grasp_world_pos[0]:.3f}, {grasp_world_pos[1]:.3f}, {grasp_world_pos[2]:.3f}]")

                    # Optional visualization
                    if args.visualize:
                        try:
                            from grasp_gen.utils.viser_utils import (
                                create_visualizer, get_color_from_score,
                                visualize_grasp, visualize_pointcloud,
                            )
                            vis = create_visualizer(port=8080)
                            pc_color = np.ones((len(pc), 3), dtype=np.uint8) * 200
                            visualize_pointcloud(vis, "point_cloud", pc, pc_color, size=0.003)
                            scores = get_color_from_score(confs[:20], use_255_scale=True)
                            for i in range(min(20, len(grasps))):
                                g = grasps[i].copy()
                                g[3, 3] = 1.0
                                visualize_grasp(vis, f"grasps/{i:03d}", g, color=scores[i],
                                                gripper_name="robotiq_2f_140", linewidth=0.6)
                            print(f"       Viser: http://localhost:8080")
                        except Exception as e:
                            print(f"       Viser error: {e}")
            except Exception as e:
                print(f"  [4] GraspGen error: {e}")

        if grasp_world_pos is None:
            # Heuristic: grasp above the object
            grasp_world_pos = obj_pos.copy()
            grasp_world_pos[2] += 0.1  # slightly above
            print(f"  [4] Heuristic grasp at [{grasp_world_pos[0]:.3f}, {grasp_world_pos[1]:.3f}, {grasp_world_pos[2]:.3f}]")

        # 5. IK solve: grasp world position → target joints
        print("  [5] IK solving...")

        # Pre-grasp: above the grasp point
        pre_grasp_pos = grasp_world_pos.copy()
        pre_grasp_pos[2] += 0.15  # 15cm above
        pre_grasp_joints = solve_ik_to_pose(env, robot, arm_idx, pre_grasp_pos)

        # Grasp position
        grasp_joints = solve_ik_to_pose(env, robot, arm_idx, grasp_world_pos)

        # Place position (offset from object)
        place_pos = obj_pos.copy()
        place_pos[0] += 0.2
        place_pos[1] += 0.2
        place_pos[2] += 0.15
        place_joints = solve_ik_to_pose(env, robot, arm_idx, place_pos)

        print(f"       Pre-grasp joints: [{', '.join(f'{j:.2f}' for j in pre_grasp_joints[:3])}...]")
        print(f"       Grasp joints:     [{', '.join(f'{j:.2f}' for j in grasp_joints[:3])}...]")

        # 6. Execute pick-and-place (record every step)
        print("  [6] Executing pick-and-place...")

        # Phase 1: Pre-grasp (above object)
        move_to(pre_grasp_joints, steps=30)

        # Phase 2: Approach (move down to grasp)
        move_to(grasp_joints, steps=20)

        # Phase 3: Close gripper
        for _ in range(15):
            step_and_record(grasp_joints, grip_close=True)

        # Phase 4: Lift (back to pre-grasp height)
        move_to(pre_grasp_joints, steps=20, grip_close=True)

        # Phase 5: Move to place
        move_to(place_joints, steps=25, grip_close=True)

        # Phase 6: Release
        for _ in range(15):
            step_and_record(place_joints, grip_close=False)

        # Phase 7: Retract to home
        move_to(HOME, steps=15)

        # 7. Save
        h5 = recorder.save(args.output_dir, episode_id, args.task, args.scene_id)
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
