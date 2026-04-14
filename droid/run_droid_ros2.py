"""
DROID env (IsaacLab) + ROS2 bridge publishing.

Runs the IsaacLab DROID environment (Franka+Robotiq+3cam) with keyboard teleop,
and publishes all data over ROS2 for external data collection.

Published topics:
  /isaac_joint_states    (sensor_msgs/JointState)
  /isaac_joint_commands  (sensor_msgs/JointState) - subscribe for MoveIt control
  /clock                 (rosgraph_msgs/Clock)
  /cam_ext1/rgb          (sensor_msgs/Image)
  /cam_ext2/rgb          (sensor_msgs/Image)
  /cam_wrist/rgb         (sensor_msgs/Image)

Usage:
    cd ~/proj/IsaacLab
    conda activate env_isaaclab
    bash isaaclab.sh -p ~/proj/isaacsim-ros2-moveit/droid_sim/run_droid_ros2.py
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DROID IsaacLab + ROS2")
parser.add_argument("--scene_id", type=str, default="1", choices=["1", "2", "3"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Cameras are in scene (for ROS2 OmniGraph publishing), so this flag is required
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports after sim init ──
import torch
import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import usdrt.Sdf

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from envs.droid_env import DroidEnvCfg

# Enable ROS2 bridge + MCP extension
import omni.kit.app
_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
# Enable MCP extension dependencies first, then MCP itself
for dep in ["omni.isaac.core", "omni.isaac.nucleus"]:
    try:
        _ext_mgr.set_extension_enabled_immediate(dep, True)
    except Exception:
        pass
_mcp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp_extension")
_ext_mgr.add_path(_mcp_path)
_ext_mgr.set_extension_enabled_immediate("isaac.sim.mcp_extension", True)
print("[MCP] Extension enabled on localhost:8766")


def setup_ros2_graph(env):
    """Create ROS2 OmniGraph to publish joints, clock, and cameras."""

    # ── Joint states + clock + MoveIt control ──
    og.Controller.edit(
        {"graph_path": "/World/ROS2_Graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("CamExt1", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CamExt2", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CamWrist", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("OnTick.outputs:tick", "CamExt1.inputs:execIn"),
                ("OnTick.outputs:tick", "CamExt2.inputs:execIn"),
                ("OnTick.outputs:tick", "CamWrist.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("Context.outputs:context", "CamExt1.inputs:context"),
                ("Context.outputs:context", "CamExt2.inputs:context"),
                ("Context.outputs:context", "CamWrist.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", "/World/envs/env_0/robot"),
                ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path("/World/envs/env_0/robot")]),
                # TF for robot + all scene objects
                ("PublishTF.inputs:targetPrims", [
                    usdrt.Sdf.Path("/World/envs/env_0/robot"),
                    usdrt.Sdf.Path("/World/envs/env_0/scene"),
                ]),
            ],
        },
    )

    # ── Camera render products ──
    cam_map = {
        "CamExt1": "/World/envs/env_0/external_cam_1",
        "CamExt2": "/World/envs/env_0/external_cam_2",
        "CamWrist": "/World/envs/env_0/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    }
    topic_map = {
        "CamExt1": "cam_ext1/rgb",
        "CamExt2": "cam_ext2/rgb",
        "CamWrist": "cam_wrist/rgb",
    }

    for node_name, cam_path in cam_map.items():
        try:
            rp = rep.create.render_product(cam_path, resolution=(320, 240))
            og.Controller.set(
                og.Controller.attribute(f"/World/ROS2_Graph/{node_name}.inputs:renderProductPath"),
                str(rp.path),
            )
            og.Controller.set(
                og.Controller.attribute(f"/World/ROS2_Graph/{node_name}.inputs:topicName"),
                topic_map[node_name],
            )
            og.Controller.set(
                og.Controller.attribute(f"/World/ROS2_Graph/{node_name}.inputs:type"),
                "rgb",
            )
            og.Controller.set(
                og.Controller.attribute(f"/World/ROS2_Graph/{node_name}.inputs:frameId"),
                node_name.lower(),
            )
            print(f"  Camera ROS2 publisher: {topic_map[node_name]} from {cam_path}")
        except Exception as e:
            print(f"  Warning: Failed to setup {node_name}: {e}")

    print("ROS2 OmniGraph created (OnTick auto-fires every render)")


def main():
    # ── Environment ──
    env_cfg = DroidEnvCfg()
    env_cfg.set_scene(args.scene_id)
    env_cfg.terminations.time_out = None
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── ROS2 bridge ──
    print("\nSetting up ROS2 bridge...")
    setup_ros2_graph(env)

    # ── Keyboard teleop ──
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05))

    should_reset = False

    def on_reset():
        nonlocal should_reset
        should_reset = True
        print("[Teleop] Reset triggered")

    teleop.add_callback("R", on_reset)

    # ── Init ──
    env.reset()
    teleop.reset()

    print("\n" + "=" * 60)
    print("DROID IsaacLab + ROS2 Bridge")
    print("=" * 60)
    print("Teleop: W/S=-X/+X  A/D=-Y/+Y  Q/E=Z  Z/X=roll  T/G=pitch  C/V=yaw")
    print("        K=gripper  R=reset")
    print("ROS2:   /isaac_joint_states  /cam_ext1/rgb  /cam_ext2/rgb  /cam_wrist/rgb")
    print("=" * 60 + "\n")

    # ── Main loop (OmniGraph ticked via physx callback registered in setup_ros2_graph) ──
    while simulation_app.is_running():
        with torch.inference_mode():
            action = teleop.advance()
            # Invert W/S (X axis, index 0) and A/D (Y axis, index 1)
            action[0] = -action[0]
            action[1] = -action[1]
            actions = action.unsqueeze(0).repeat(env.num_envs, 1)
            env.step(actions)

            if should_reset:
                env.reset()
                teleop.reset()
                should_reset = False

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
