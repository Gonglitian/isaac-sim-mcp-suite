# MCP Command Reference

The Isaac Sim MCP extension exposes a TCP JSON API on `localhost:8766`. Commands are sent as JSON objects with `type` and `params` fields. Responses contain a `status` field (`"success"` or `"error"`) and command-specific data.

All examples below use `mcp_client.py`:

```python
from scripts.mcp_client import IsaacMCP
mcp = IsaacMCP(host="localhost", port=8766)
```

---

## Core Commands

### get_scene_info

Get basic scene information and verify connectivity.

**Parameters**: None

**Returns**:
- `status` (str): `"success"` if connected
- `message` (str): `"pong"`
- `assets_root_path` (str): Isaac Sim Nucleus assets root path

**Example**:
```python
result = mcp.get_scene_info()
# {"status": "success", "result": {"status": "success", "message": "pong",
#  "assets_root_path": "omniverse://localhost/NVIDIA/Assets/Isaac/4.2"}}
```

### execute_script

Execute arbitrary Python code inside the Isaac Sim process. This is the most powerful command -- it has full access to all Isaac Sim, Omniverse, and USD Python APIs.

**Parameters**:
- `code` (str, required): Python source code to execute. The namespace includes `omni`, `carb`, `Usd`, `UsdGeom`, `Sdf`, `Gf`.

**Returns**:
- `status` (str): `"success"` or `"error"`
- `message` (str): Status message or error description
- `traceback` (str, on error): Full Python traceback

**Example**:
```python
result = mcp.execute("""
import omni.kit.commands
omni.kit.commands.execute('CreatePrim', prim_type='Cube')
print("Cube created")
""")
```

### create_robot

Spawn a robot from the Isaac Sim asset library.

**Parameters**:
- `robot_type` (str, default `"franka"`): One of `"franka"`, `"jetbot"`, `"carter"`, `"g1"`, `"go1"`. Unknown types default to Franka.
- `position` (list[float], default `[0, 0, 0]`): World position `[x, y, z]`.

**Returns**:
- `status` (str): `"success"` or `"error"`
- `message` (str): Confirmation message

**Example**:
```python
result = mcp.create_robot(robot_type="franka", position=[0, 0, 0])
```

### create_physics_scene

Create a physics-enabled scene with a ground plane and primitive objects.

**Parameters**:
- `objects` (list[dict], default `[]`): List of object descriptors. Each dict can contain:
  - `name` (str): Object name
  - `type` (str): `"Cube"`, `"Sphere"`, `"Cylinder"`, `"Cone"`, `"Plane"`
  - `path` (str): USD prim path (auto-generated if omitted)
  - `position` (list[float]): `[x, y, z]`
  - `rotation` (list[float]): Quaternion `[w, x, y, z]`, default `[1, 0, 0, 0]`
  - `scale` (list[float]): `[sx, sy, sz]`, default `[1, 1, 1]`
  - `color` (list[float]): `[r, g, b, a]`, default `[0.5, 0.5, 0.5, 1.0]`
  - `physics_enabled` (bool): default `True`
  - `mass` (float): default `1.0`
  - `is_kinematic` (bool): default `False`
  - `size` (float): Primitive size, default `100.0`
- `floor` (bool, default `True`): Create a ground plane
- `gravity` (list[float], default `[0, -9.81, 0]`): Gravity vector
- `scene_name` (str, default `"None"`): Scene identifier

**Returns**:
- `status` (str): `"success"` or `"error"`
- `message` (str): Count of objects created

**Example**:
```python
result = mcp.create_physics_scene(
    objects=[
        {"name": "red_cube", "type": "Cube", "position": [0, 0, 0.5],
         "scale": [0.1, 0.1, 0.1], "color": [1, 0, 0, 1]},
        {"name": "blue_sphere", "type": "Sphere", "position": [0.3, 0, 0.5],
         "scale": [0.05, 0.05, 0.05], "color": [0, 0, 1, 1]},
    ],
    floor=True,
)
```

### omni_kit_command

Execute a named Omniverse Kit command. Note: the MCP handler is registered as `omini_kit_command` (with a typo in the extension source).

**Parameters**:
- `command` (str, required): Kit command name, e.g. `"CreatePrim"`
- `prim_type` (str, required): Prim type argument, e.g. `"Cube"`

**Returns**:
- `status` (str): `"success"` or `"error"`

**Example**:
```python
result = mcp._send("omini_kit_command", {"command": "CreatePrim", "prim_type": "Sphere"})
```

### transform

Set the position and scale of any USD prim.

**Parameters**:
- `prim_path` (str, required): USD prim path, e.g. `"/World/Cube"`
- `position` (list[float], default `[0, 0, 50]`): World position `[x, y, z]`
- `scale` (list[float], default `[10, 10, 10]`): Scale `[sx, sy, sz]`

**Returns**:
- `status` (str): `"success"` or `"error"`
- `position` (list): Applied position
- `scale` (list): Applied scale

**Example**:
```python
result = mcp.transform(prim_path="/World/Cube", position=[1, 0, 0.5], scale=[0.1, 0.1, 0.1])
```

---

## Data Collection Commands

### get_all_poses

Get world-space poses of all Xform prims under a root path.

**Parameters**:
- `root_path` (str, default `"/World"`): Root path to search under

**Returns**:
- `status` (str): `"success"` or `"error"`
- `poses` (dict): `{prim_path: {"position": [x,y,z], "orientation": [w,x,y,z]}}` for each prim
- `count` (int): Number of prims found

**Example**:
```python
result = mcp._send("get_all_poses", {"root_path": "/World/envs/env_0/scene"})
poses = result["result"]["poses"]
for path, pose in poses.items():
    print(f"{path}: pos={pose['position']}")
```

### get_robot_state

Get robot joint positions, velocities, and end-effector pose.

**Parameters**:
- `robot_path` (str, default `"/World/envs/env_0/robot"`): USD path to the robot articulation root

**Returns**:
- `status` (str): `"success"` or `"error"`
- `joints` (dict): `{joint_name: {"target": float}}` for each joint with a drive
- `ee_pose` (dict or null): `{"position": [x,y,z], "orientation": [w,x,y,z], "frame": str}` for end-effector
- `robot_path` (str): Echoed robot path

**Example**:
```python
result = mcp._send("get_robot_state", {"robot_path": "/World/envs/env_0/robot"})
joints = result["result"]["joints"]
ee = result["result"]["ee_pose"]
print(f"EE position: {ee['position']}")
```

### sim_control

Control the simulation timeline: play, pause, stop, step, reset, or query status.

**Parameters**:
- `action` (str, default `"status"`): One of `"play"`, `"pause"`, `"stop"`, `"step"`, `"reset"`, `"status"`
- `num_steps` (int, default `1`): Number of frames to advance (only for `"step"` action)

**Returns**:
- `status` (str): `"success"` or `"error"`
- `message` (str): Action confirmation
- (for `"status"` action) `playing` (bool), `stopped` (bool), `current_time` (float)

**Example**:
```python
mcp._send("sim_control", {"action": "play"})
mcp._send("sim_control", {"action": "step", "num_steps": 10})
state = mcp._send("sim_control", {"action": "status"})
print(f"Playing: {state['result']['playing']}")
```

### screenshot

Capture a screenshot from the viewport or a specific camera prim.

**Parameters**:
- `camera_path` (str, default `"viewport"`): `"viewport"` for the main viewport, or a camera prim path (e.g. `"/World/cam_ext1"`)
- `save_path` (str, default `"/tmp/mcp_screenshot.png"`): Output file path
- `width` (int, default `640`): Image width (camera prim only)
- `height` (int, default `480`): Image height (camera prim only)

**Returns**:
- `status` (str): `"success"` or `"error"`
- `save_path` (str): Where the file was written
- `source` (str): `"viewport"` or the resolved camera prim path

**Example**:
```python
result = mcp._send("screenshot", {
    "camera_path": "/World/cam_ext1",
    "save_path": "/tmp/ext1.png"
})
```

---

## Scene Diversity / Domain Randomization Commands

### spawn_object

Spawn a primitive shape or USD asset into the scene.

**Parameters**:
- `obj_type` (str, default `"Cube"`): `"Cube"`, `"Sphere"`, `"Cylinder"`, or any USD prim type
- `name` (str, default `"object"`): Name of the prim (placed at `/World/envs/env_0/{name}`)
- `position` (list[float], default `[0, 0, 0.5]`): World position `[x, y, z]`
- `scale` (list[float], default `[0.05, 0.05, 0.05]`): Scale `[sx, sy, sz]`
- `color` (list[float], default `[0.5, 0.5, 0.5]`): RGB color `[r, g, b]` (ignored if `usd_path` is set)
- `physics` (bool, default `True`): Apply RigidBodyAPI and CollisionAPI
- `usd_path` (str, default `""`): Path to a USD file to load instead of a primitive

**Returns**:
- `status` (str): `"success"` or `"error"`
- `prim_path` (str): Full USD path of the spawned object

**Example**:
```python
result = mcp._send("spawn_object", {
    "obj_type": "Sphere",
    "name": "ball",
    "position": [0.3, 0, 0.5],
    "scale": [0.03, 0.03, 0.03],
    "color": [1, 0, 0],
    "physics": True,
})
```

### delete_object

Remove a prim from the USD stage.

**Parameters**:
- `prim_path` (str, required): Full USD path to delete, e.g. `"/World/envs/env_0/ball"`

**Returns**:
- `status` (str): `"success"` or `"error"`
- `message` (str): Confirmation

**Example**:
```python
result = mcp._send("delete_object", {"prim_path": "/World/envs/env_0/ball"})
```

### randomize_scene

Apply domain randomization to object positions, lighting, and/or colors.

**Parameters**:
- `randomize_objects` (bool, default `True`): Randomize positions of scene objects (skips table, lights, physics materials)
- `randomize_lighting` (bool, default `True`): Randomize light intensity (2000--8000) and color
- `randomize_colors` (bool, default `False`): Assign random diffuse colors to scene objects
- `object_pos_range` (list[list[float]], default `[[-0.3,-0.3,0],[0.3,0.3,0]]`): `[min_xyz, max_xyz]` position range
- `root_path` (str, default `"/World/envs/env_0/scene"`): Root prim whose children get randomized

**Returns**:
- `status` (str): `"success"` or `"error"`
- `changes` (list[str]): Human-readable list of changes applied
- `count` (int): Number of changes

**Example**:
```python
result = mcp._send("randomize_scene", {
    "randomize_objects": True,
    "randomize_lighting": True,
    "randomize_colors": True,
    "object_pos_range": [[-0.2, -0.2, 0.4], [0.2, 0.2, 0.6]],
})
for change in result["result"]["changes"]:
    print(change)
```

---

## Save / Load / Joint Control Commands

### save_scene

Export the current USD stage to a file.

**Parameters**:
- `file_path` (str, default `""`): Output path. If empty, saves to the current stage's file path or `/tmp/saved_scene.usd`.

**Returns**:
- `status` (str): `"success"` or `"error"`
- `file_path` (str): Where the scene was saved

**Example**:
```python
result = mcp._send("save_scene", {"file_path": "/tmp/my_scene.usd"})
```

### load_scene

Open a USD stage file.

**Parameters**:
- `file_path` (str, required): Path to the USD file to load

**Returns**:
- `status` (str): `"success"` or `"error"`
- `file_path` (str): Echoed file path

**Example**:
```python
result = mcp._send("load_scene", {"file_path": "/tmp/my_scene.usd"})
```

### set_robot_joints

Set robot joint drive targets directly by name.

**Parameters**:
- `robot_path` (str, default `"/World/envs/env_0/robot"`): USD path to the robot
- `joint_positions` (dict, default `{}`): `{joint_name: value_in_radians}` or `{joint_name: value_in_meters}` for prismatic joints

**Returns**:
- `status` (str): `"success"` or `"error"`
- `set_joints` (list[str]): Names of joints that were set
- `message` (str): Count of joints set

**Example**:
```python
result = mcp._send("set_robot_joints", {
    "robot_path": "/World/envs/env_0/robot",
    "joint_positions": {
        "panda_joint1": 0.0,
        "panda_joint2": -0.628,
        "panda_joint3": 0.0,
        "panda_joint4": -2.513,
        "panda_joint5": 0.0,
        "panda_joint6": 1.885,
        "panda_joint7": 0.0,
        "finger_joint": 0.0,
    }
})
```

---

## Communication Protocol

All commands use raw TCP sockets on port 8766. The protocol is:

1. Client opens a TCP connection to `localhost:8766`
2. Client sends a single JSON message: `{"type": "<command_name>", "params": {<param_dict>}}`
3. Server executes the command in Isaac Sim's main thread via `run_coroutine`
4. Server sends back a single JSON response: `{"status": "success"|"error", "result": {...}}`
5. Connection is closed after each command

The `IsaacMCP` client class in `scripts/mcp_client.py` wraps this protocol with convenience methods. For commands not wrapped by `IsaacMCP`, use `mcp._send(cmd_type, params)` directly.
