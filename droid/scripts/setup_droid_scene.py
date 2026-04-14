"""
Setup DROID scene in standalone Isaac Sim via MCP.

Creates: Franka+Robotiq 2F-85, 3 cameras, table, ROS2 OmniGraph.
Publishes: /clock, /isaac_joint_states, /isaac_joint_commands,
           /cam_ext1/rgb, /cam_ext2/rgb, /cam_wrist/rgb

Usage:
    python droid_sim/scripts/setup_droid_scene.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "moveit", "scripts"))

from mcp_client import IsaacMCP
import json
import pathlib

# Asset directory (downloaded from HuggingFace, see README)
ASSET_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "assets")

mcp = IsaacMCP()
if not mcp.is_connected():
    print("ERROR: Cannot connect to Isaac Sim MCP (localhost:8766)")
    print("Start Isaac Sim first: ./launch_isaacsim.sh")
    sys.exit(1)

print("Connected to Isaac Sim.\n")

# ── Step 1: Load Franka+Robotiq + Table + Lighting ──
print("[Step 1] Loading scene...")
result = mcp.execute(f'''
import numpy as np
import omni.kit.commands
from isaacsim.core.utils import prims, rotations, stage, viewports
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux
import isaacsim.core.experimental.utils.stage as stage_utils

# Fresh stage
stage_utils.create_new_stage(template="sunlight")
assets_root_path = get_assets_root_path()

# Camera view
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# Franka + Robotiq 2F-85 (from sim-evals USD)
robot_usd = "{ASSET_DIR}/franka_robotiq_2f_85_flattened.usd"
robot = prims.create_prim(
    "/World/robot", "Xform",
    position=np.array([0, 0, 0]),
    usd_path=robot_usd,
)
print("Loaded Franka + Robotiq 2F-85")

# Table
table_usd = "{ASSET_DIR}/table.usd"
table = prims.create_prim(
    "/World/table", "Xform",
    position=np.array([0.5, 0, 0]),
    usd_path=table_usd,
)
print("Loaded table")

# Ground plane
from isaacsim.core.api.objects import GroundPlane
GroundPlane("/World/ground", visible=True)
print("Scene loaded: Franka+Robotiq + table + ground")
''')
print(f"  Result: {result.get('status')}")

# ── Step 2: Create 3 cameras ──
print("\n[Step 2] Creating 3 DROID cameras...")
result = mcp.execute(r'''
from pxr import UsdGeom, Sdf, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

cam_configs = [
    {
        "name": "cam_ext1",
        "path": "/World/cam_ext1",
        "pos": Gf.Vec3d(0.05, 0.57, 0.66),
        "focal": 21.0,
    },
    {
        "name": "cam_ext2",
        "path": "/World/cam_ext2",
        "pos": Gf.Vec3d(0.05, -0.57, 0.66),
        "focal": 21.0,
    },
    {
        "name": "cam_wrist",
        "path": "/World/robot/Gripper/Robotiq_2F_85/base_link/cam_wrist",
        "pos": Gf.Vec3d(0.011, -0.031, -0.074),
        "focal": 28.0,
    },
]

for cfg in cam_configs:
    cam_prim = stage.DefinePrim(cfg["path"], "Camera")
    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(cfg["focal"])
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(cfg["pos"])
    print(f"  Created camera: {cfg['name']} at {cfg['path']}")

print("3 DROID cameras created")
''')
print(f"  Result: {result.get('status')}")

# ── Step 3: ROS2 OmniGraph (joints + clock + cameras + MoveIt) ──
print("\n[Step 3] Creating ROS2 OmniGraph...")
result = mcp.execute(r'''
import omni.graph.core as og
import usdrt.Sdf

og.Controller.edit(
    {"graph_path": "/World/ROS2_Graph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            # Joint states for MoveIt
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            # Clock
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            # Camera helpers (3 cameras)
            ("CamExt1Helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CamExt2Helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CamWristHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ],
        og.Controller.Keys.CONNECT: [
            # Tick all on impulse
            ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "CamExt1Helper.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "CamExt2Helper.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "CamWristHelper.inputs:execIn"),
            # Context
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("Context.outputs:context", "CamExt1Helper.inputs:context"),
            ("Context.outputs:context", "CamExt2Helper.inputs:context"),
            ("Context.outputs:context", "CamWristHelper.inputs:context"),
            # Timestamps
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            # MoveIt joint command → articulation controller
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        og.Controller.Keys.SET_VALUES: [
            # Joint state topics (MoveIt compatible)
            ("ArticulationController.inputs:robotPath", "/World/robot"),
            ("PublishJointState.inputs:topicName", "isaac_joint_states"),
            ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
            ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path("/World/robot")]),
            # Camera topics
            ("CamExt1Helper.inputs:topicName", "cam_ext1/rgb"),
            ("CamExt1Helper.inputs:type", "rgb"),
            ("CamExt1Helper.inputs:cameraPrim", [usdrt.Sdf.Path("/World/cam_ext1")]),
            ("CamExt1Helper.inputs:frameId", "cam_ext1"),
            ("CamExt2Helper.inputs:topicName", "cam_ext2/rgb"),
            ("CamExt2Helper.inputs:type", "rgb"),
            ("CamExt2Helper.inputs:cameraPrim", [usdrt.Sdf.Path("/World/cam_ext2")]),
            ("CamExt2Helper.inputs:frameId", "cam_ext2"),
            ("CamWristHelper.inputs:topicName", "cam_wrist/rgb"),
            ("CamWristHelper.inputs:type", "rgb"),
            ("CamWristHelper.inputs:cameraPrim", [usdrt.Sdf.Path("/World/robot/Gripper/Robotiq_2F_85/base_link/cam_wrist")]),
            ("CamWristHelper.inputs:frameId", "cam_wrist"),
        ],
    },
)
print("ROS2 OmniGraph created:")
print("  Joints: /isaac_joint_states, /isaac_joint_commands")
print("  Clock:  /clock")
print("  Cams:   /cam_ext1/rgb, /cam_ext2/rgb, /cam_wrist/rgb")
''')
print(f"  Result: {result.get('status')}")

# ── Step 4: Init physics + start sim + tick callback ──
print("\n[Step 4] Starting simulation...")
result = mcp.execute(r'''
from isaacsim.core.api import SimulationContext
import omni.graph.core as og
import omni.physx
import builtins

sim_ctx = SimulationContext(stage_units_in_meters=1.0)
sim_ctx.initialize_physics()
sim_ctx.play()

# Tick OmniGraph every physics step
def tick_graph(dt):
    og.Controller.set(
        og.Controller.attribute("/World/ROS2_Graph/OnImpulseEvent.state:enableImpulse"), True
    )

builtins._graph_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(tick_graph)
print("Simulation playing, OmniGraph ticking every physics step")
''')
print(f"  Result: {result.get('status')}")

print("\n" + "=" * 60)
print("DROID Scene Ready!")
print("=" * 60)
print("ROS2 topics:")
print("  /isaac_joint_states  /isaac_joint_commands  /clock")
print("  /cam_ext1/rgb  /cam_ext2/rgb  /cam_wrist/rgb")
print()
print("Next steps:")
print("  1. Launch MoveIt:  ros2 launch ~/proj/isaacsim-ros2-moveit/launch_moveit.py")
print("  2. Start recorder: python droid_sim/scripts/ros2_data_collector.py")
print("=" * 60)
