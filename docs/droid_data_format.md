# DROID HDF5 Data Format

This document describes the HDF5 trajectory format used for DROID-compatible data collection in this project.

## Episode Directory Structure

Each recorded episode produces a directory:

```
droid_data/
  episode_0000/
    trajectory.h5       # All observation/action data
    metadata.json       # Episode metadata (human-readable)
  episode_0001/
    trajectory.h5
    metadata.json
  ...
```

## HDF5 File Schema (trajectory.h5)

### File-Level Attributes

| Attribute | Type | Description |
|---|---|---|
| `success` | bool | Whether the episode was marked successful |
| `num_steps` | int | Total number of timesteps (T) |
| `task_description` | str | Natural language task instruction |
| `control_frequency_hz` | float | Control loop frequency (15.0 Hz) |

Note: The `scene_id` attribute is present in trajectories recorded via `DroidRecorder` (IsaacLab direct) but may be absent in trajectories from the ROS2 data collector.

### Full Dataset Tree

```
trajectory.h5
|
|-- [attrs] success: bool
|-- [attrs] num_steps: int
|-- [attrs] task_description: str
|-- [attrs] control_frequency_hz: float
|
|-- observation/
|   |-- robot_state/
|   |   |-- joint_positions          (T, 7)    float64   # panda_joint1..7 in radians
|   |   |-- joint_velocities         (T, 7)    float64   # joint velocities in rad/s
|   |   |-- gripper_position         (T,)      float64   # normalized [0=open, 1=closed]
|   |   +-- cartesian_position       (T, 6)    float64   # [x,y,z,roll,pitch,yaw] (DroidRecorder only)
|   |
|   |-- camera_images/
|   |   |-- exterior_image_1_left    (T, 180, 320, 3)  uint8   # RGB, gzip compressed
|   |   |-- exterior_image_2_left    (T, 180, 320, 3)  uint8   # RGB, gzip compressed
|   |   +-- wrist_image_left         (T, 180, 320, 3)  uint8   # RGB, gzip compressed
|   |
|   +-- object_poses/               (ROS2 data collector only)
|       +-- <object_name>/
|           |-- position             (T, 3)    float64   # [x, y, z] world frame
|           +-- orientation          (T, 4)    float64   # [qx, qy, qz, qw] quaternion
|
|-- action/
|   |-- joint_position              (T, 7)    float64   # target arm joint positions
|   |-- gripper_position            (T,)      float64   # target gripper [0=open, 1=closed]
|   |-- cartesian_position          (T, 6)    float64   # EE pose (DroidRecorder only)
|   |-- cartesian_velocity          (T, 6)    float64   # EE velocity (DroidRecorder only)
|   |-- joint_velocity              (T, 7)    float64   # joint velocity (DroidRecorder only)
|   +-- gripper_velocity            (T,)      float64   # gripper velocity (DroidRecorder only)
|
|-- action_flat                     (T, 7)    float64   # [cartesian_pos(6), gripper(1)] (DroidRecorder only)
|
+-- timestamps                      (T,)      float64   # wall-clock time (ROS2 collector only)
```

### Camera Images

All camera images are stored as raw uint8 RGB arrays at a fixed resolution of **180 x 320** pixels (height x width). Images are captured from three camera viewpoints:

| Camera | Topic (ROS2) | Mounting | IsaacLab Prim Path |
|---|---|---|---|
| exterior_image_1_left | /cam_ext1/rgb | Fixed, left side | {ENV}/external_cam_1 |
| exterior_image_2_left | /cam_ext2/rgb | Fixed, right side | {ENV}/external_cam_2 |
| wrist_image_left | /cam_wrist/rgb | Attached to gripper base_link | {ENV}/robot/.../wrist_cam |

Camera parameters (IsaacLab):
- External cameras: focal_length=2.1mm, horizontal_aperture=5.376mm, vertical_aperture=3.024mm
- Wrist camera: focal_length=2.8mm, same aperture

Images are resized to 180x320 from the native resolution (720x1280 in IsaacLab, 320x240 via ROS2 OmniGraph) before storage. Compression uses gzip level 4.

### Object Poses

When recording via the ROS2 data collector (`ros2_data_collector.py`), object poses are extracted from the `/tf` topic. Robot links (containing `panda_`, `finger`, or `knuckle` in the frame ID) are filtered out. Only scene objects are recorded.

Object names are sanitized: `/` characters are replaced with `_` and leading underscores are stripped. For example, TF frame `World_envs_env_0_scene_cube` maps to HDF5 group `observation/object_poses/World_envs_env_0_scene_cube/`.

### Gripper Normalization

The gripper position is normalized to the range [0, 1]:
- **0.0** = fully open (finger_joint = 0.0 rad)
- **1.0** = fully closed (finger_joint = pi/4 rad)

The conversion formula: `gripper_normalized = finger_joint_rad / (pi / 4)`

## Metadata JSON (metadata.json)

```json
{
  "task_description": "put the cube in the bowl",
  "episode_id": 0,
  "num_steps": 656,
  "success": true,
  "control_frequency_hz": 15.0,
  "duration_s": 43.73
}
```

Additional fields may be present depending on the recorder:
- `scene_id` (str): Scene variant ("1", "2", or "3")
- `start_time` / `end_time` (float): Unix timestamps (DroidRecorder only)

## Two Recording Paths

There are two ways to record DROID trajectories in this project. They produce slightly different HDF5 schemas:

### 1. DroidRecorder (IsaacLab direct)

Used by `droid_sim/collect_data.py`. Records inside the IsaacLab process. Produces the full schema including `cartesian_position`, `cartesian_velocity`, `joint_velocity`, `gripper_velocity`, and `action_flat`. Does NOT include `timestamps` or `object_poses`.

### 2. ROS2 Data Collector (external node)

Used by `droid_sim/scripts/ros2_data_collector.py`. Runs as a standalone ROS2 node. Produces `timestamps` and `object_poses` groups. Does NOT include velocity-based action fields, `cartesian_position` in observations, or `action_flat`.

## Comparison with Real DROID Format

The real-world DROID dataset (https://droid-dataset.github.io/) stores data in RLDS/TFRecord format, not HDF5. Key differences:

| Aspect | Real DROID | This Project (Sim) |
|---|---|---|
| Storage format | RLDS (TFRecord) | HDF5 |
| Image storage | Encoded video frames | Raw uint8 arrays (gzip) |
| Image resolution | 180x320 | 180x320 (matched) |
| Number of cameras | 2 external + 1 wrist | 2 external + 1 wrist (matched) |
| Camera naming | exterior_image_{1,2}_left | exterior_image_{1,2}_left (matched) |
| Robot | Franka + Robotiq 2F-85 | Franka + Robotiq 2F-85 (matched) |
| Action space | SE(3) delta + gripper | IK-relative SE(3) delta + binary gripper |
| Joint names | panda_joint1..7 | panda_joint1..7 (matched) |
| Control frequency | 15 Hz | 15 Hz (matched) |
| Gripper convention | 0=open, 1=closed | 0=open, 1=closed (matched) |
| Object poses | Not included | Included (via TF, sim only) |
| Task instruction | Natural language | Natural language (matched) |

The sim-evals project (upstream) has validated that policies trained on real DROID data can transfer zero-shot to this simulation environment, confirming that the observation space and robot configuration are sufficiently compatible.

## Reading Trajectories

```python
import h5py
import numpy as np

with h5py.File("droid_data/episode_0000/trajectory.h5", "r") as f:
    # Metadata
    print(f"Task: {f.attrs['task_description']}")
    print(f"Steps: {f.attrs['num_steps']}, Success: {f.attrs['success']}")

    # Robot state
    joint_pos = f["observation/robot_state/joint_positions"][:]    # (T, 7)
    gripper = f["observation/robot_state/gripper_position"][:]     # (T,)

    # Camera images
    ext1 = f["observation/camera_images/exterior_image_1_left"][:] # (T, 180, 320, 3)

    # Actions
    action_jp = f["action/joint_position"][:]                      # (T, 7)
    action_grip = f["action/gripper_position"][:]                  # (T,)

    # Object poses (if recorded via ROS2 collector)
    if "observation/object_poses" in f:
        for obj_name in f["observation/object_poses"]:
            pos = f[f"observation/object_poses/{obj_name}/position"][:]
            ori = f[f"observation/object_poses/{obj_name}/orientation"][:]
            print(f"Object {obj_name}: {pos.shape}")
```
