---
name: isaac-mcp-guide
description: Guide for controlling NVIDIA Isaac Sim via MCP extension. Use when the user asks to interact with Isaac Sim, create scenes, spawn objects, control robots, collect data, or take screenshots in simulation.
---

# Isaac Sim MCP Control Guide

## 1. MCP TCP Protocol

The Isaac Sim MCP extension listens on **localhost:8766** via a raw TCP socket. Communication is JSON-based:

```
Send: {"type": "command_name", "params": {...}}
Recv: {"status": "success"|"error", "result": {...}}
```

### Python Client

Located at `moveit/scripts/mcp_client.py`:

```python
from scripts.mcp_client import IsaacMCP

mcp = IsaacMCP(host="localhost", port=8766, timeout=30.0)

# Verify connection
mcp.is_connected()        # returns True/False
mcp.get_scene_info()      # returns scene hierarchy

# Run arbitrary Python inside Isaac Sim
mcp.execute("print('hello from sim')")

# Spawn a robot
mcp.create_robot(robot_type="franka", position=[0, 0, 0])

# Create a physics scene with objects
mcp.create_physics_scene(objects=["cube", "sphere"], floor=True)

# Move/scale an object
mcp.transform("/World/Cube", position=[1, 0, 0.5], scale=[0.5, 0.5, 0.5])
```

The client opens a new TCP connection per command. For raw socket usage:

```python
import socket, json

def send_mcp(cmd_type, params=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 8766))
    s.settimeout(30.0)
    s.sendall(json.dumps({"type": cmd_type, "params": params or {}}).encode())
    result = json.loads(s.recv(65536).decode())
    s.close()
    return result
```

---

## 2. All MCP Commands

### get_scene_info

Get the current scene hierarchy. Use to verify connection and inspect the stage.

- **Params:** none
- **Returns:** scene tree with prim paths and types

```python
mcp.get_scene_info()
# or
send_mcp("get_scene_info")
```

### execute_script

Run arbitrary Python code inside the Isaac Sim process. The most powerful command -- anything you can do in Isaac Sim's Script Editor, you can do here.

- **Params:** `code` (str) -- Python code to execute
- **Returns:** stdout output and any return value

```python
mcp.execute("""
import omni.usd
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    print(prim.GetPath())
""")
```

### get_all_poses

Get positions and orientations of all objects under a root path.

- **Params:** `root_path` (str) -- USD prim path to search under (e.g., "/World")
- **Returns:** dict mapping prim paths to position/orientation

```python
send_mcp("get_all_poses", {"root_path": "/World"})
```

### get_robot_state

Query a robot's current joint positions, end-effector pose, and gripper state.

- **Params:** `robot_path` (str) -- prim path of the robot (e.g., "/World/envs/env_0/robot")
- **Returns:** joint_positions, ee_pose, gripper_state

```python
send_mcp("get_robot_state", {"robot_path": "/World/envs/env_0/robot"})
```

### sim_control

Control the simulation timeline.

- **Params:** `action` (str) -- one of: "play", "pause", "stop", "step", "reset", "status"; `num_steps` (int, optional) -- number of steps for "step" action
- **Returns:** current simulation status

```python
send_mcp("sim_control", {"action": "play"})
send_mcp("sim_control", {"action": "step", "num_steps": 100})
send_mcp("sim_control", {"action": "status"})
```

### screenshot

Capture an image from the viewport or a specific camera.

- **Params:** `camera_path` (str, optional) -- prim path of camera; `save_path` (str, optional) -- file path to save the image
- **Returns:** image data or confirmation of save

```python
send_mcp("screenshot", {
    "camera_path": "/World/envs/env_0/external_cam_1",
    "save_path": "/tmp/scene_capture.png"
})
```

### spawn_object

Add a primitive or USD asset to the scene.

- **Params:**
  - `obj_type` (str) -- "cube", "sphere", "cylinder", "cone", "plane", or "custom"
  - `name` (str) -- prim name
  - `position` (list[3]) -- [x, y, z] world coordinates
  - `scale` (list[3], optional) -- [sx, sy, sz] scale factors
  - `color` (list[3], optional) -- [r, g, b] in 0-1 range
  - `physics` (bool, optional) -- enable rigid body physics
  - `usd_path` (str, optional) -- path to USD asset (for obj_type="custom")
- **Returns:** prim path of created object

```python
send_mcp("spawn_object", {
    "obj_type": "cube",
    "name": "RedCube",
    "position": [0.5, 0, 0.5],
    "scale": [0.05, 0.05, 0.05],
    "color": [1, 0, 0],
    "physics": True
})

# Spawn a USD asset
send_mcp("spawn_object", {
    "obj_type": "custom",
    "name": "mug",
    "position": [0.3, 0.1, 0.8],
    "usd_path": "/path/to/mug.usd",
    "physics": True
})
```

### delete_object

Remove a prim from the stage.

- **Params:** `prim_path` (str) -- full prim path to delete
- **Returns:** confirmation

```python
send_mcp("delete_object", {"prim_path": "/World/RedCube"})
```

### randomize_scene

Apply domain randomization to the current scene.

- **Params:**
  - `randomize_objects` (bool) -- randomize object poses
  - `randomize_lighting` (bool) -- randomize light properties
  - `randomize_colors` (bool) -- randomize material colors
- **Returns:** confirmation with randomization details

```python
send_mcp("randomize_scene", {
    "randomize_objects": True,
    "randomize_lighting": True,
    "randomize_colors": True
})
```

### save_scene

Export the current stage to a USD file.

- **Params:** `file_path` (str) -- output file path (e.g., "/tmp/my_scene.usd")
- **Returns:** confirmation

```python
send_mcp("save_scene", {"file_path": "/tmp/my_scene.usd"})
```

### load_scene

Open a USD file as the current stage.

- **Params:** `file_path` (str) -- path to USD file
- **Returns:** confirmation

```python
send_mcp("load_scene", {"file_path": "/tmp/my_scene.usd"})
```

### set_robot_joints

Set a robot's joint positions directly (bypasses motion planning).

- **Params:** `robot_path` (str) -- prim path; `joint_positions` (list[float]) -- target joint angles in radians
- **Returns:** confirmation

```python
send_mcp("set_robot_joints", {
    "robot_path": "/World/envs/env_0/robot",
    "joint_positions": [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
})
```

### create_robot

Spawn a preset robot from the built-in library.

- **Params:** `robot_type` (str) -- one of "franka", "jetbot", "g1", "go1"; `position` (list[3]) -- spawn location
- **Returns:** prim path of created robot

```python
mcp.create_robot(robot_type="franka", position=[0, 0, 0])
mcp.create_robot(robot_type="g1", position=[2, 0, 0])
```

### transform

Move and/or scale an existing prim.

- **Params:** `prim_path` (str) -- prim to transform; `position` (list[3], optional) -- new position; `scale` (list[3], optional) -- new scale
- **Returns:** confirmation

```python
mcp.transform("/World/Cube", position=[1, 0, 0.5], scale=[2, 2, 2])
```

### create_physics_scene

Create a physics-enabled world with optional objects and ground plane.

- **Params:**
  - `objects` (list[str], optional) -- primitive types to spawn (e.g., ["cube", "sphere"])
  - `floor` (bool, optional) -- add a ground plane (default True)
  - `gravity` (list[3], optional) -- gravity vector (default [0, 0, -9.81])
- **Returns:** confirmation with created prim paths

```python
mcp.create_physics_scene(
    objects=["cube", "sphere", "cylinder"],
    floor=True
)
```

---

## 3. Isaac Sim Python API Reference (for execute_script)

Use these APIs when writing code to send via `execute_script`. All code runs inside Isaac Sim's Python 3.11 interpreter.

### Stage & Scene

```python
import omni.usd
stage = omni.usd.get_context().get_stage()

# Create a fresh stage with default lighting
import isaacsim.core.experimental.utils.stage as stage_utils
stage_utils.create_new_stage(template="sunlight")

# Load a USD asset onto the stage
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path="path/to/asset.usd", prim_path="/World/MyAsset")

# Get the assets root path (for NVIDIA-provided assets)
from isaacsim.storage.native import get_assets_root_path
assets_root = get_assets_root_path()
```

### USD/Pxr (Geometry, Transforms, Physics)

```python
from pxr import Gf, UsdGeom, UsdPhysics, Sdf, UsdShade

# Get a prim and make it transformable
prim = stage.GetPrimAtPath("/World/Cube")
xform = UsdGeom.Xformable(prim)

# Set position
xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.5))

# Set rotation (euler angles in degrees)
xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 45))

# Set scale
xform.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1))

# Enable physics on a prim
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)

# Set mass
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr(0.5)
```

### Prims

```python
import numpy as np
from isaacsim.core.utils import prims

# Create a prim with position and optional USD reference
prim = prims.create_prim(
    prim_path="/World/obj",
    prim_type="Xform",
    position=np.array([0, 0, 1]),
    usd_path="path/to/asset.usd"
)

# Check if a prim exists
exists = prims.is_prim_path_valid("/World/obj")

# Delete a prim
prims.delete_prim("/World/obj")
```

### OmniGraph (ROS2 Bridge)

Create OmniGraph nodes to bridge Isaac Sim data to ROS2 topics:

```python
import omni.graph.core as og
import usdrt.Sdf

og.Controller.edit(
    {"graph_path": "/World/Graph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("PublishJointState.inputs:topicName", "/isaac_joint_states"),
            ("SubscribeJointState.inputs:topicName", "/isaac_joint_commands"),
            ("ArticulationController.inputs:robotPath", "/World/envs/env_0/robot"),
            ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path("/World/envs/env_0/robot")]),
        ],
    }
)
```

**Common ROS2 OmniGraph node types:**

| Node Type | Purpose |
|---|---|
| `omni.graph.action.OnTick` | Fires every simulation tick (use this, NOT OnImpulseEvent) |
| `isaacsim.ros2.bridge.ROS2PublishJointState` | Publish joint states to ROS2 |
| `isaacsim.ros2.bridge.ROS2SubscribeJointState` | Receive joint commands from ROS2 |
| `isaacsim.ros2.bridge.ROS2CameraHelper` | Publish camera images to ROS2 |
| `isaacsim.ros2.bridge.ROS2PublishClock` | Publish sim clock to /clock |
| `isaacsim.ros2.bridge.ROS2PublishTransformTree` | Publish TF transforms |
| `isaacsim.core.nodes.IsaacArticulationController` | Apply joint commands to robot |

### Simulation

```python
from isaacsim.core.api import SimulationContext

sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

# Step the simulation
for _ in range(100):
    sim.step(render=True)

sim.pause()
sim.stop()
```

### Viewports

```python
import numpy as np
from isaacsim.core.utils import viewports

# Set the camera view in the viewport
viewports.set_camera_view(
    eye=np.array([1.2, 1.2, 0.8]),
    target=np.array([0, 0, 0.5])
)
```

---

## 4. DROID Environment Reference

The DROID environment is the primary teleoperation and data-collection setup.

### Robot Configuration

- **Robot:** Franka Panda + Robotiq 2F-85 gripper
- **Prim path:** `/World/envs/env_0/robot`
- **Arm joints:** `panda_joint1` through `panda_joint7` (7-DOF)
- **Gripper:** `finger_joint` -- 0 = fully open, pi/4 (~0.785) = fully closed

### Cameras

| Camera | Prim Name | Resolution |
|---|---|---|
| External camera 1 | `external_cam_1` | 720 x 1280 |
| External camera 2 | `external_cam_2` | 720 x 1280 |
| Wrist camera | `wrist_cam` | 720 x 1280 |

### ROS2 Topics

| Topic | Type | Direction |
|---|---|---|
| `/isaac_joint_states` | sensor_msgs/JointState | Isaac Sim -> ROS2 |
| `/isaac_joint_commands` | sensor_msgs/JointState | ROS2 -> Isaac Sim |
| `/cam_ext1/rgb` | sensor_msgs/Image | Isaac Sim -> ROS2 |
| `/cam_ext2/rgb` | sensor_msgs/Image | Isaac Sim -> ROS2 |
| `/cam_wrist/rgb` | sensor_msgs/Image | Isaac Sim -> ROS2 |
| `/clock` | rosgraph_msgs/Clock | Isaac Sim -> ROS2 |
| `/tf` | tf2_msgs/TFMessage | Isaac Sim -> ROS2 |

### Control Parameters

- **Control frequency:** 15 Hz
- **Physics dt:** 1/120 s
- **Decimation:** 8 (physics steps per control step)

### Scene Objects (scene1)

- `rubiks_cube` -- Rubik's cube on the table
- `_24_bowl` -- bowl
- `table` -- workspace table

---

## 5. Common Gotchas

### Python Version Mismatch
Isaac Sim uses Python 3.11; system ROS2 (Humble) uses Python 3.10. **Never source the ROS2 setup.bash before launching Isaac Sim.** Launch Isaac Sim first, then source ROS2 in a separate terminal.

### RMW Implementation
Both Isaac Sim and ROS2 nodes must use the same RMW:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```
Set this in every terminal before launching any ROS2 node or Isaac Sim.

### Persistent State in execute_script
Each `execute_script` call runs in a fresh exec() context. To persist variables across calls, attach them to builtins:
```python
mcp.execute("""
import builtins
builtins.my_controller = SomeController()
""")

# Later call can access it:
mcp.execute("""
import builtins
builtins.my_controller.step()
""")
```

### OmniGraph Tick Nodes in IsaacLab
In IsaacLab environments, always use `omni.graph.action.OnTick` which fires automatically every simulation step. Do NOT use `omni.graph.action.OnImpulseEvent` -- it requires manual triggering and will not fire during normal simulation.

### Camera Buffer Race Conditions
When using IsaacLab's observation framework to read camera data, buffer race conditions can occur (stale or partially-written frames). Instead, publish camera images via ROS2 OmniGraph nodes (`ROS2CameraHelper`) and subscribe on the ROS2 side.

### Assets
NVIDIA-provided robot and object assets need to be downloaded separately. Community and custom assets are available on HuggingFace. Use `get_assets_root_path()` for built-in assets, or provide absolute file paths for custom USD files.
