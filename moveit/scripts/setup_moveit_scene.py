"""
Setup Isaac Sim scene for MoveIt2 integration via MCP.

Creates:
- Franka Panda robot at origin
- Simple room environment
- MoveIt-compatible OmniGraph (publish /isaac_joint_states, subscribe /isaac_joint_commands)
- Physics tick callback

Run this AFTER Isaac Sim is launched with MCP extension.

Usage:
    python scripts/setup_moveit_scene.py
"""
from mcp_client import IsaacMCP
import json
import sys

mcp = IsaacMCP()

if not mcp.is_connected():
    print("ERROR: Cannot connect to Isaac Sim MCP (localhost:8766)")
    print("Make sure Isaac Sim is running with MCP extension enabled.")
    sys.exit(1)

print("Connected to Isaac Sim.")

# Step 1: Create scene with Franka + environment
print("\n[Step 1] Setting up scene...")
result = mcp.execute(r'''
import numpy as np
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.utils import prims, rotations, stage, viewports
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf
import isaacsim.core.experimental.utils.stage as stage_utils

# Create fresh stage with sunlight
stage_utils.create_new_stage(template="sunlight")

assets_root_path = get_assets_root_path()

# Camera
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# Environment
stage.add_reference_to_stage(
    assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd", "/background"
)

# Franka robot
robot = prims.create_prim(
    "/Franka", "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

print("Scene loaded: Franka + Simple_Room")
''')
print(f"  Result: {result.get('status')}")

# Step 2: Create MoveIt-compatible OmniGraph
print("\n[Step 2] Creating ROS2 OmniGraph...")
result = mcp.execute(r'''
import omni.graph.core as og
import usdrt.Sdf

og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ArticulationController.inputs:robotPath", "/Franka"),
            ("PublishJointState.inputs:topicName", "isaac_joint_states"),
            ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
            ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path("/Franka")]),
        ],
    },
)
print("OmniGraph created: /isaac_joint_states, /isaac_joint_commands, /clock")
''')
print(f"  Result: {result.get('status')}")

# Step 3: Initialize physics and start simulation
print("\n[Step 3] Starting simulation...")
result = mcp.execute(r'''
from isaacsim.core.api import SimulationContext
sim_ctx = SimulationContext(stage_units_in_meters=1.0)
sim_ctx.initialize_physics()
sim_ctx.play()
print("Simulation playing")
''')
print(f"  Result: {result.get('status')}")

# Step 4: Register tick callback for OmniGraph
print("\n[Step 4] Registering OmniGraph tick callback...")
result = mcp.execute(r'''
import omni.graph.core as og
import omni.physx
import builtins

def tick_graph(dt):
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

builtins._graph_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(tick_graph)
print("Tick callback registered")
''')
print(f"  Result: {result.get('status')}")

print("\n" + "=" * 50)
print("Isaac Sim ready for MoveIt2!")
print("ROS2 topics: /isaac_joint_states, /isaac_joint_commands, /clock")
print()
print("Next: Launch MoveIt2 in another terminal:")
print("  source /opt/ros/humble/setup.bash")
print("  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
print("  ros2 launch ~/proj/isaacsim-ros2-moveit/launch_moveit.py")
print("=" * 50)
