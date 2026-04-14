# System Architecture

## High-Level System Diagram

```
+===========================================================================+
||  GPU Host (Ubuntu 22.04, NVIDIA RTX)                                    ||
||                                                                          ||
||  +------------------------------+    +--------------------------------+ ||
||  |  Isaac Sim 5.1               |    |  IsaacLab 2.3.0               | ||
||  |  (Standalone Application)    |    |  (Python Library via conda)   | ||
||  |                              |    |                                | ||
||  |  +--PhysX 5 (GPU physics)   |    |  +--ManagerBasedRLEnv         | ||
||  |  +--Hydra Renderer           |    |  +--DroidEnvCfg               | ||
||  |  +--USD Stage                |    |  +--DifferentialIK            | ||
||  |  +--OmniGraph                |    |  +--BinaryGripperAction       | ||
||  |  +--Kit Extensions:          |    |  +--3 CameraCfg sensors       | ||
||  |  |  - MCP Extension (:8766) |    |  +--Event/Obs/Term managers   | ||
||  |  |  - ROS2 Bridge            |    |                                | ||
||  |  |  - Replicator             |    +----------+---------------------+ ||
||  |  +--Nucleus Asset Server     |               |                       ||
||  +--------+-----+---------------+               |                       ||
||           |     |                               |                       ||
||   TCP     |     | DDS                           | Direct Python         ||
||   :8766   |     | (FastRTPS)                    | (isaaclab.sh -p)      ||
||           |     |                               |                       ||
||  +--------v-+   |   +---------------------------v---------------------+ ||
||  | MCP      |   |   |  Data Collection Layer                          | ||
||  | Client   |   |   |                                                 | ||
||  | Scripts  |   |   |  collect_data.py      DroidRecorder             | ||
||  |          |   |   |  (keyboard teleop) -> (HDF5 writer)             | ||
||  | setup_*  |   |   |                                                 | ||
||  | .py      |   |   |  run_droid_ros2.py                              | ||
||  +----------+   |   |  (teleop + ROS2      OmniGraph bridge           | ||
||                 |   |   OmniGraph setup)    (topics below)             | ||
||                 |   +-------------------------------------------------+ ||
||                 |                                                        ||
||    +------------v-----------------------------------------------+       ||
||    |  ROS2 Humble (System /opt/ros/humble)                      |       ||
||    |                                                             |       ||
||    |  Topics:                                                    |       ||
||    |    /isaac_joint_states    sensor_msgs/JointState  (pub)    |       ||
||    |    /isaac_joint_commands  sensor_msgs/JointState  (sub)    |       ||
||    |    /clock                 rosgraph_msgs/Clock     (pub)    |       ||
||    |    /cam_ext1/rgb          sensor_msgs/Image       (pub)    |       ||
||    |    /cam_ext2/rgb          sensor_msgs/Image       (pub)    |       ||
||    |    /cam_wrist/rgb         sensor_msgs/Image       (pub)    |       ||
||    |    /tf                    tf2_msgs/TFMessage       (pub)    |       ||
||    |    /joint_states          sensor_msgs/JointState  (remap)  |       ||
||    |    /collector/cmd         std_msgs/String          (sub)    |       ||
||    |                                                             |       ||
||    |  Nodes:                                                     |       ||
||    |    move_group             (MoveIt2 planner)                 |       ||
||    |    rviz2                  (visualization + interactive)     |       ||
||    |    ros2_control_node      (topic_based_ros2_control)        |       ||
||    |    robot_state_publisher  (URDF -> TF)                      |       ||
||    |    static_tf_publisher    (world -> panda_link0)            |       ||
||    |    droid_data_collector   (15 Hz HDF5 recorder)             |       ||
||    +-------------------------------------------------------------+       ||
||                                                                          ||
||  +-------------------------------------------------------------------+  ||
||  |  Output: droid_data/episode_NNNN/                                 |  ||
||  |    trajectory.h5    (obs + action + images + object_poses)        |  ||
||  |    metadata.json    (task, success, duration, hz)                  |  ||
||  +-------------------------------------------------------------------+  ||
+===========================================================================+
```

## MCP Communication Flow

The MCP (Model Context Protocol) extension provides remote control of Isaac Sim over TCP.

```
MCP Client (Python)                     Isaac Sim MCP Extension
       |                                         |
       |  1. TCP connect to localhost:8766       |
       |---------------------------------------->|
       |                                         |
       |  2. Send JSON command                   |
       |  {"type": "execute_script",             |
       |   "params": {"code": "..."}}            |
       |---------------------------------------->|
       |                                         |
       |     3. Extension receives on             |
       |        daemon socket thread              |
       |                                         |
       |     4. Dispatches to handler via         |
       |        run_coroutine() to execute        |
       |        on Kit's main event loop          |
       |                                         |
       |     5. Handler executes in Isaac Sim     |
       |        context (full USD/PhysX access)   |
       |                                         |
       |  6. JSON response                        |
       |  {"status": "success",                   |
       |   "result": {...}}                       |
       |<-----------------------------------------|
       |                                         |
       |  7. TCP connection closed                |
       |                                         |
```

Key design details:
- Each command opens a new TCP connection (no persistent sessions)
- Commands are dispatched to named handlers in `_execute_command_internal()`
- The `execute_script` command runs arbitrary Python via `exec()` in a namespace pre-loaded with `omni`, `carb`, `Usd`, `UsdGeom`, `Sdf`, `Gf`
- Socket buffer is 16384 bytes per recv; responses up to 65536 bytes
- The server thread is a daemon thread that exits when Isaac Sim shuts down

## ROS2 Topic List and Data Flow

### Topics Published by Isaac Sim (via OmniGraph)

| Topic | Message Type | Rate | Source |
|---|---|---|---|
| `/isaac_joint_states` | sensor_msgs/JointState | Physics rate | ROS2PublishJointState node |
| `/isaac_joint_commands` | sensor_msgs/JointState | -- (subscribed) | ROS2SubscribeJointState node |
| `/clock` | rosgraph_msgs/Clock | Physics rate | ROS2PublishClock node |
| `/cam_ext1/rgb` | sensor_msgs/Image | Render rate | ROS2CameraHelper node |
| `/cam_ext2/rgb` | sensor_msgs/Image | Render rate | ROS2CameraHelper node |
| `/cam_wrist/rgb` | sensor_msgs/Image | Render rate | ROS2CameraHelper node |
| `/tf` | tf2_msgs/TFMessage | Physics rate | ROS2PublishTransformTree node |

### MoveIt2 Data Flow

```
RViz (user drags interactive marker)
  |
  v
MoveIt2 move_group (OMPL/CHOMP/Pilz planning)
  |
  v
FollowJointTrajectory action server
  |
  v
ros2_control_node (topic_based_ros2_control)
  |
  v  publishes to:
/isaac_joint_commands (sensor_msgs/JointState)
  |
  v  OmniGraph subscribes:
ROS2SubscribeJointState -> ArticulationController
  |
  v
Robot moves in simulation (PhysX)
  |
  v  OmniGraph publishes:
ROS2PublishJointState -> /isaac_joint_states
  |
  v
ros2_control_node reads joint feedback
  |
  v  remapped to:
/joint_states
  |
  v
robot_state_publisher -> TF
  |
  v
RViz (synchronized robot display)
```

### Data Collection Data Flow (ROS2 Path)

```
Isaac Sim / IsaacLab
  |
  | OmniGraph (OnTick or OnImpulseEvent)
  |
  +---> /isaac_joint_states  --+
  +---> /cam_ext1/rgb         |
  +---> /cam_ext2/rgb         +---> ros2_data_collector.py
  +---> /cam_wrist/rgb        |       (latest-value pattern, 15 Hz timer)
  +---> /tf                  --+       |
                                       v
                                  DroidDataCollector node
                                       |
                                       v
                                  trajectory.h5 + metadata.json
```

## IsaacLab Environment Internals

### DroidEnvCfg (ManagerBasedRLEnvCfg)

The DROID environment uses IsaacLab's Manager-Based RL Environment pattern:

```
DroidEnvCfg
  |
  +-- scene: DroidSceneCfg
  |     +-- robot: FRANKA_ROBOTIQ_CFG (ArticulationCfg)
  |     |     +-- Franka Panda 7-DOF arm
  |     |     +-- Robotiq 2F-85 gripper (finger_joint)
  |     |     +-- ImplicitActuatorCfg for shoulder, forearm, gripper
  |     +-- external_cam_1: CameraCfg (1280x720, opengl convention)
  |     +-- external_cam_2: CameraCfg (1280x720, opengl convention)
  |     +-- wrist_cam: CameraCfg (1280x720, attached to gripper base_link)
  |     +-- sphere_light: AssetBaseCfg
  |     +-- scene_env: loaded from scene{1,2,3}.usd via load_scene()
  |
  +-- actions: ActionCfg
  |     +-- arm: DifferentialInverseKinematicsActionCfg
  |     |     command_type="pose", use_relative_mode=True
  |     |     body_name="panda_link8", ik_method="dls"
  |     +-- gripper: BinaryGripperActionCfg
  |           0 = open (0.0 rad), 1 = close (pi/4 rad)
  |
  +-- observations: ObservationCfg
  |     +-- policy:
  |           +-- arm_joint_pos: (7,) panda_joint1..7
  |           +-- gripper_pos: (1,) normalized [0,1]
  |           (Camera images NOT in obs pipeline -- published via ROS2)
  |
  +-- events: EventCfg
  |     +-- reset_all: reset_scene_to_default on reset
  |
  +-- terminations: TerminationsCfg
  |     +-- time_out: (disabled during teleop)
  |
  +-- timing:
        episode_length_s = 600
        decimation = 8
        sim.dt = 1/120
        control frequency = 120/8 = 15 Hz
```

### Action Space

The environment accepts a 7D action tensor per step:

```
action = [dx, dy, dz, rx, ry, rz, gripper]
           |--- SE(3) delta (IK) ---|  |-- binary: >0.5 = close
```

The DifferentialIKController converts the 6D SE(3) delta into 7 target joint positions using damped least squares (DLS). The BinaryGripperAction maps the last dimension to finger_joint open/close commands.

### Scene Variants

Three pre-built DROID scenes are available as USD files:

| Scene ID | Objects | Task |
|---|---|---|
| 1 | Cube + Bowl | "put the cube in the bowl" |
| 2 | Can + Mug | (manipulation) |
| 3 | Banana + Bin | (manipulation) |

Scenes are loaded from `droid_sim/assets/scene{1,2,3}.usd`, originally from the sim-evals dataset on HuggingFace (`owhan/DROID-sim-environments`).

### Robot Configuration (FRANKA_ROBOTIQ_CFG)

```
ArticulationCfg:
  USD: franka_robotiq_2f_85_flattened.usd
  Gravity: disabled
  Self-collisions: disabled
  Solver iterations: position=64, velocity=0

  Initial joint positions:
    panda_joint1: 0.0
    panda_joint2: -pi/5
    panda_joint3: 0.0
    panda_joint4: -4*pi/5
    panda_joint5: 0.0
    panda_joint6: 3*pi/5
    panda_joint7: 0.0
    finger_joint: 0.0  (open)

  Actuators:
    panda_shoulder (joints 1-4): stiffness=400, damping=80, effort_limit=87
    panda_forearm  (joints 5-7): stiffness=400, damping=80, effort_limit=12
    gripper        (finger_joint): velocity_limit=1.0
```

## Data Collection Pipeline

### Pipeline 1: IsaacLab Direct (collect_data.py)

```
Se3Keyboard (device)
  |
  | 7D action [dx,dy,dz,rx,ry,rz,gripper]
  v
ManagerBasedRLEnv.step(actions)
  |
  | Physics: 8 sub-steps at 1/120s = 1/15s per control step
  | IK solver: delta -> target joint positions
  | Gripper: binary threshold -> open/close command
  |
  v
Read robot state:
  - joint_pos from robot.data.joint_pos (arm indices)
  - joint_vel from robot.data.joint_vel
  - gripper from finger_joint / (pi/4)
  - ee_pose from robot.data.body_pos_w[-1]
  |
  v
DroidRecorder.record_timestep()
  |
  | In-memory buffer (list of dicts)
  |
  v (on SPACE=success or ESC=failure)
DroidRecorder.end_episode()
  |
  | Write HDF5: observation/robot_state/*, observation/camera_images/*,
  |             action/*, action_flat
  | Write JSON: metadata.json
  v
droid_data/episode_NNNN/
```

### Pipeline 2: ROS2 External (run_droid_ros2.py + ros2_data_collector.py)

```
Se3Keyboard (device)                    Isaac Sim Process
  |                                      |
  | 7D action                            |
  v                                      |
ManagerBasedRLEnv.step()                |
  |                                      |
  v                                      |
OmniGraph (setup_ros2_graph)            |
  |                                      |
  +---> /isaac_joint_states   -----DDS---->  ros2_data_collector.py
  +---> /cam_ext1/rgb         -----DDS---->    |
  +---> /cam_ext2/rgb         -----DDS---->    | latest-value subscribers
  +---> /cam_wrist/rgb        -----DDS---->    | + 15 Hz timer
  +---> /tf                   -----DDS---->    |
                                               v
                                          _record_timer_cb()
                                            - extract arm joints (panda_joint1..7)
                                            - normalize gripper
                                            - decode images via cv_bridge
                                            - filter TF for scene objects
                                            - append to timesteps[]
                                               |
                                               v (on keyboard 'n' or /collector/cmd)
                                          save_episode()
                                            - write HDF5
                                            - write metadata.json
                                               |
                                               v
                                          droid_data/episode_NNNN/
```

### Pipeline Comparison

| Feature | Pipeline 1 (Direct) | Pipeline 2 (ROS2) |
|---|---|---|
| Runs inside IsaacLab process | Yes | Recorder is external |
| Camera access | Direct tensor read | ROS2 Image messages |
| Object poses | Not recorded | Recorded from /tf |
| Velocity actions | Computed inline | Not recorded |
| Cartesian EE pose | From body_pos_w | Not recorded |
| action_flat field | Yes | No |
| Timestamps | Not recorded | Wall-clock time |
| MoveIt compatible | No | Yes (can use MoveIt for teleop) |
| Typical FPS | ~15 Hz (control) | ~15 Hz (timer-based) |

### OmniGraph Node Configuration

The ROS2 OmniGraph is constructed programmatically in `run_droid_ros2.py` and `setup_droid_scene.py`. The graph uses either `OnTick` (IsaacLab path, auto-fires every render frame) or `OnImpulseEvent` + physx callback (standalone Isaac Sim path, manually pulsed each physics step).

```
OmniGraph: /World/ROS2_Graph (execution evaluator)

Tick Source -------> PublishJointState ------> /isaac_joint_states
    |   +----------> SubscribeJointState ----> ArticulationController
    |   +----------> PublishClock -----------> /clock
    |   +----------> PublishTF --------------> /tf
    |   +----------> CamExt1 ----------------> /cam_ext1/rgb
    |   +----------> CamExt2 ----------------> /cam_ext2/rgb
    |   +----------> CamWrist ---------------> /cam_wrist/rgb
    |
ROS2Context ------> (shared by all ROS2 nodes)
ReadSimTime ------> (timestamps for JointState and Clock)
```
