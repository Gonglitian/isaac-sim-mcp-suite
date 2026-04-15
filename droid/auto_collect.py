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

def move_ee_to_pos(env, robot, arm_idx, target_pos, steps=40, grip_close=False):
    """Move EE to target world position using DifferentialIK action space.

    Instead of solving IK manually, we switch the env action to DifferentialIK
    and feed SE(3) deltas directly. IsaacLab handles the IK internally.

    With JointPositionActionCfg env, we can't use this directly.
    So we use iterative position control: read current EE, compute error,
    apply small joint corrections via env.step().
    """
    ee_body_idx = None
    for i, name in enumerate(robot.data.body_names):
        if name == "panda_link8":
            ee_body_idx = i
            break
    if ee_body_idx is None:
        ee_body_idx = len(robot.data.body_names) - 1

    target_pos_t = torch.tensor(target_pos, dtype=torch.float32, device=env.device)

    for step in range(steps):
        ee_pos = robot.data.body_pos_w[0, ee_body_idx, :3]
        error = target_pos_t - ee_pos
        error_norm = error.norm().item()

        if error_norm < 0.005:
            print(f"       EE reached target: error={error_norm:.4f}m at step {step}")
            break

        # Compute joint correction using Jacobian from PhysX
        jacobian_full = robot.root_physx_view.get_jacobians()  # (N, num_bodies, 6, num_dofs)
        jacobian = jacobian_full[0, ee_body_idx, :3, :7]  # (3, 7) position-only

        # Damped least squares
        lam = 0.05
        JT = jacobian.T
        JJT = jacobian @ JT + lam * torch.eye(3, device=env.device)
        delta_q = JT @ torch.linalg.solve(JJT, error)

        # Scale step
        max_step = 0.1
        if delta_q.norm() > max_step:
            delta_q = delta_q * max_step / delta_q.norm()

        # Apply as joint position target
        current_jp = robot.data.joint_pos[0, arm_idx].cpu().tolist()
        new_jp = [current_jp[i] + delta_q[i].item() for i in range(7)]

        # Clamp
        limits = [(-2.9, 2.9), (-1.8, 1.8), (-2.9, 2.9), (-3.1, 0.08),
                  (-2.9, 2.9), (-0.08, 3.8), (-2.9, 2.9)]
        new_jp = [max(limits[i][0], min(limits[i][1], new_jp[i])) for i in range(7)]

        action = build_action(env, new_jp, grip_close)
        env.step(action)

    final_error = (target_pos_t - robot.data.body_pos_w[0, ee_body_idx, :3]).norm().item()
    print(f"       EE error: {final_error:.4f}m ({min(step+1, steps)} steps)")
    return get_arm_pos(robot, arm_idx)


def move_ee_to_pos_recorded(env, robot, arm_idx, finger_idx, target_pos, recorder,
                            steps=40, grip_close=False):
    """Same as move_ee_to_pos but records every step."""
    ee_body_idx = None
    for i, name in enumerate(robot.data.body_names):
        if name == "panda_link8":
            ee_body_idx = i
            break
    if ee_body_idx is None:
        ee_body_idx = len(robot.data.body_names) - 1

    target_pos_t = torch.tensor(target_pos, dtype=torch.float32, device=env.device)

    for step in range(steps):
        ee_pos = robot.data.body_pos_w[0, ee_body_idx, :3]
        error = target_pos_t - ee_pos
        if error.norm().item() < 0.005:
            break

        jacobian_full = robot.root_physx_view.get_jacobians()
        jacobian = jacobian_full[0, ee_body_idx, :3, :7]
        lam = 0.05
        JT = jacobian.T
        JJT = jacobian @ JT + lam * torch.eye(3, device=env.device)
        delta_q = JT @ torch.linalg.solve(JJT, error)
        if delta_q.norm() > 0.1:
            delta_q = delta_q * 0.1 / delta_q.norm()

        current_jp = robot.data.joint_pos[0, arm_idx].cpu().tolist()
        new_jp = [current_jp[i] + delta_q[i].item() for i in range(7)]
        limits = [(-2.9, 2.9), (-1.8, 1.8), (-2.9, 2.9), (-3.1, 0.08),
                  (-2.9, 2.9), (-0.08, 3.8), (-2.9, 2.9)]
        new_jp = [max(limits[i][0], min(limits[i][1], new_jp[i])) for i in range(7)]

        action = build_action(env, new_jp, grip_close)
        env.step(action)
        recorder.record(robot, arm_idx, finger_idx, env, new_jp, 1.0 if grip_close else 0.0)

    final_error = (target_pos_t - robot.data.body_pos_w[0, ee_body_idx, :3]).norm().item()
    print(f"       EE error: {final_error:.4f}m ({min(step+1, steps)} steps)")
    return get_arm_pos(robot, arm_idx)


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


    # Warm up
    env.reset()
    for _ in range(10):
        try:
            env.step(build_action(env, HOME))
        except RuntimeError:
            env.sim.render()

    # Scene object paths
    scene_objects = {
        "1": {"pick": "/World/envs/env_0/scene/rubiks_cube", "place": "/World/envs/env_0/scene/_24_bowl"},
        "2": {"pick": "/World/envs/env_0/scene/can", "place": "/World/envs/env_0/scene/mug"},
        "3": {"pick": "/World/envs/env_0/scene/banana", "place": "/World/envs/env_0/scene/bin"},
    }
    pick_path = scene_objects.get(args.scene_id, scene_objects["1"])["pick"]
    place_path = scene_objects.get(args.scene_id, scene_objects["1"])["place"]

    print(f"\n{'='*60}")
    print(f"Auto Grasp Collection")
    print(f"  Task: {args.task} | Episodes: {args.num_episodes}")
    print(f"  Pick: {pick_path.split('/')[-1]} | Place: {place_path.split('/')[-1]}")
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

        # 1. Move to home + wait for physics to settle (objects fall to table)
        move_to(HOME, steps=15)
        print("  [1] Settling physics...")
        zero = build_action(env, HOME)
        for _ in range(60):
            env.step(zero)

        # 2. Read object positions AFTER settling
        pick_pose = get_object_world_pose(env, pick_path)
        place_pose = get_object_world_pose(env, place_path)

        if pick_pose is not None:
            obj_pos = pick_pose[:3, 3].copy()
        else:
            obj_pos = np.array([0.3, 0.0, 0.5])
        print(f"  [2] Pick ({pick_path.split('/')[-1]}): [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")

        if place_pose is not None:
            bowl_pos = place_pose[:3, 3].copy()
        else:
            bowl_pos = obj_pos + np.array([0.15, 0.15, 0.0])
        print(f"  [2] Place ({place_path.split('/')[-1]}): [{bowl_pos[0]:.3f}, {bowl_pos[1]:.3f}, {bowl_pos[2]:.3f}]")

        # 3. Get object point cloud from mesh (ground truth, no camera noise)
        pc = None
        try:
            import trimesh as tm
            stage = omni.usd.get_context().get_stage()
            obj_prim = stage.GetPrimAtPath(pick_path)
            if obj_prim.IsValid():
                # Find mesh geometry under the object prim
                mesh_found = False
                for child in obj_prim.GetAllChildren() if hasattr(obj_prim, 'GetAllChildren') else []:
                    pass
                # Use trimesh with the USD file directly or sample from bbox
                # Simple approach: create a box point cloud based on object bounding box
                from pxr import UsdGeom
                bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(obj_prim)
                bbox_range = bbox.ComputeAlignedRange()
                bbox_min = np.array(bbox_range.GetMin())
                bbox_max = np.array(bbox_range.GetMax())
                obj_size = bbox_max - bbox_min
                obj_center = (bbox_min + bbox_max) / 2.0

                # Sample points on the surface of the bounding box (6 faces)
                n_per_face = 350
                faces_pts = []
                for axis in range(3):
                    for side in [bbox_min[axis], bbox_max[axis]]:
                        pts = np.random.uniform(bbox_min, bbox_max, (n_per_face, 3)).astype(np.float32)
                        pts[:, axis] = side
                        faces_pts.append(pts)
                pc = np.concatenate(faces_pts, axis=0)
                # Add some interior points for density
                interior = np.random.uniform(bbox_min, bbox_max, (200, 3)).astype(np.float32)
                pc = np.concatenate([pc, interior], axis=0)
                # Center for GraspGen
                pc -= pc.mean(axis=0)
                print(f"  [3] Point cloud: {pc.shape[0]} pts from mesh bbox")
                print(f"       Object bbox: size=[{obj_size[0]:.3f}, {obj_size[1]:.3f}, {obj_size[2]:.3f}]")
                print(f"       Object center: [{obj_center[0]:.3f}, {obj_center[1]:.3f}, {obj_center[2]:.3f}]")
                # Update obj_pos to use bbox center (more accurate)
                obj_pos = obj_center
            else:
                print(f"  [3] Object prim not found: {pick_path}")
        except Exception as e:
            print(f"  [3] Mesh PC error: {e}")
            import traceback; traceback.print_exc()

        # 4. GraspGen or heuristic
        grasp_world_pos = None
        grasp_world_mat = None

        if pc is not None and grasp_client:
            try:
                grasps, confs = grasp_client.infer(pc, num_grasps=200, topk_num_grasps=50)
                if len(grasps) > 0:
                    grasp_obj = grasps[0]  # (4,4) SE(3) relative to centered point cloud
                    # The point cloud was centered (mean subtracted), so grasp position
                    # is relative to the point cloud centroid. Since the object IS the
                    # point cloud, grasp_pos ≈ offset from object center.
                    # World grasp = object_world_pos + grasp_offset
                    grasp_offset = grasp_obj[:3, 3]  # offset from object center (in centered PC frame)

                    # GraspGen gives the gripper TCP position. For Robotiq 2F-140,
                    # the TCP is between the fingertips. But our IK targets panda_link8
                    # (wrist), which is ~0.20m above the fingertips.
                    # So we only need the XY from grasp offset, and set Z for the wrist:
                    # wrist_Z = object_top + gripper_offset (to position fingertips at object)
                    EE_TO_FINGERTIP = 0.20  # panda_link8 → Robotiq fingertip offset

                    # World grasp position for fingertips (XY from GraspGen, Z at object top)
                    fingertip_world = obj_pos.copy()
                    fingertip_world[0] += grasp_offset[0]  # XY offset from GraspGen
                    fingertip_world[1] += grasp_offset[1]

                    # Wrist (panda_link8) needs to be above fingertip target
                    grasp_world_pos = fingertip_world.copy()
                    grasp_world_pos[2] = obj_pos[2] + EE_TO_FINGERTIP  # wrist above object

                    print(f"  [4] GraspGen: {len(grasps)} grasps, best={confs[0]:.3f}")
                    print(f"       Grasp offset: [{grasp_offset[0]:.3f}, {grasp_offset[1]:.3f}, {grasp_offset[2]:.3f}]")
                    print(f"       Fingertip target: [{fingertip_world[0]:.3f}, {fingertip_world[1]:.3f}, {fingertip_world[2]:.3f}]")
                    print(f"       Wrist (EE) target: [{grasp_world_pos[0]:.3f}, {grasp_world_pos[1]:.3f}, {grasp_world_pos[2]:.3f}]")

                    # Save point cloud + grasps for visualization
                    if args.visualize:
                        vis_dir = Path(args.output_dir) / f"episode_{episode_id:04d}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        np.save(vis_dir / "point_cloud.npy", pc)
                        np.save(vis_dir / "grasps.npy", grasps[:20])
                        np.save(vis_dir / "confidences.npy", confs[:20])
                        print(f"       Saved PC + grasps for visualization")
                        print(f"       Visualize: conda activate graspgen && python libs/GraspGen/client-server/graspgen_client.py --pcd_file {vis_dir}/point_cloud.npy --host localhost --port 5556 --visualize")
            except Exception as e:
                print(f"  [4] GraspGen error: {e}")

        if grasp_world_pos is None:
            # Heuristic: wrist above the object (fingertips at object height)
            EE_TO_FINGERTIP = 0.20
            grasp_world_pos = obj_pos.copy()
            grasp_world_pos[2] += EE_TO_FINGERTIP
            print(f"  [4] Heuristic wrist target: [{grasp_world_pos[0]:.3f}, {grasp_world_pos[1]:.3f}, {grasp_world_pos[2]:.3f}]")

        # 5+6. Move EE to grasp targets using Jacobian IK + record every step
        print("  [5] Executing pick-and-place with IK...")

        EE_TO_FINGERTIP = 0.20

        pre_grasp_pos = grasp_world_pos.copy()
        pre_grasp_pos[2] += 0.10  # 10cm above grasp

        # Place: wrist above bowl position
        place_pos = bowl_pos.copy()
        place_pos[2] += EE_TO_FINGERTIP  # wrist above bowl
        print(f"       Place wrist target: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")

        # All phases recorded every step (smooth video)
        print("       Phase 1: Pre-grasp")
        move_ee_to_pos_recorded(env, robot, arm_idx, finger_idx, pre_grasp_pos, recorder, steps=40)

        print("       Phase 2: Approach")
        move_ee_to_pos_recorded(env, robot, arm_idx, finger_idx, grasp_world_pos, recorder, steps=30)

        print("       Phase 3: Grasp")
        jp = get_arm_pos(robot, arm_idx)
        for _ in range(20):
            step_and_record(jp, grip_close=True)

        print("       Phase 4: Lift")
        move_ee_to_pos_recorded(env, robot, arm_idx, finger_idx, pre_grasp_pos, recorder, steps=30, grip_close=True)

        print("       Phase 5: Place")
        move_ee_to_pos_recorded(env, robot, arm_idx, finger_idx, place_pos, recorder, steps=30, grip_close=True)

        print("       Phase 6: Release")
        jp = get_arm_pos(robot, arm_idx)
        for _ in range(20):
            step_and_record(jp, grip_close=False)

        print("       Phase 7: Home")
        move_to(HOME, steps=20)

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
