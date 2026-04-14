# isaac-sim-mcp-suite

Isaac Sim MCP Suite is a robotics simulation and data-collection framework that connects NVIDIA Isaac Sim 5.1 with ROS2 Humble and MoveIt2 through a TCP-based Model Context Protocol (MCP) extension. It supports three workflows: (1) interactive MoveIt2 teleoperation of a Franka Panda in Isaac Sim via RViz, (2) DROID-format data collection using an IsaacLab 2.3.0 environment with Franka + Robotiq 2F-85 and 3 cameras, and (3) standalone scene scripting through MCP commands. The system generates HDF5 trajectories compatible with the real-world DROID dataset for sim-to-real policy transfer.

## Architecture

```
+-------------------------------------------------------------------+
|  Host Machine                                                      |
|                                                                    |
|  +-----------------------+       TCP :8766        +-------------+  |
|  | Isaac Sim 5.1         |<---------------------->| MCP Client  |  |
|  | (Python 3.11 internal)|   JSON commands        | (Python 3.x)|  |
|  |  +--MCP Extension     |                        | mcp_client.py  |
|  |  +--ROS2 Bridge       |                        +-------------+  |
|  |  +--OmniGraph         |                                         |
|  |  +--PhysX             |       DDS (FastRTPS)                    |
|  +--------+--------------+<----------------------------+           |
|            |                                           |           |
|            | /isaac_joint_states                       |           |
|            | /isaac_joint_commands      +---------------+--------+  |
|            | /cam_ext1/rgb             | ROS2 Humble (system)   |  |
|            | /cam_ext2/rgb             |  MoveIt2 + RViz        |  |
|            | /cam_wrist/rgb            |  ros2_control          |  |
|            | /clock                    |  Data Collector Node   |  |
|            | /tf                       +------------------------+  |
|            |                                                       |
|  +---------v-----------+                                           |
|  | IsaacLab 2.3.0      |     (alternative to standalone Isaac Sim) |
|  | ManagerBasedRLEnv    |                                          |
|  |  DroidEnvCfg         |                                          |
|  |  Franka+Robotiq+3cam|                                          |
|  |  IK delta actions    |                                          |
|  |  15 Hz control       |                                          |
|  +---------+------------+                                          |
|            |                                                       |
|            v                                                       |
|  +---------+------------+                                          |
|  | DroidRecorder        |                                          |
|  | trajectory.h5 (HDF5) |                                          |
|  | metadata.json        |                                          |
|  +-----------------------+                                         |
+-------------------------------------------------------------------+
```

## Quick Start

### Workflow 1: MoveIt2 Interactive Teleoperation (3 terminals)

```bash
# Terminal 1 -- Isaac Sim (do NOT source ROS2 before this)
./launch_isaacsim.sh

# Terminal 2 -- Setup scene via MCP (after Isaac Sim is ready)
cd scripts && python setup_moveit_scene.py

# Terminal 3 -- MoveIt2 + RViz
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch ~/proj/isaacsim-ros2-moveit/launch_moveit.py
```

### Workflow 2: DROID Data Collection with IsaacLab

```bash
# Terminal 1 -- IsaacLab environment + keyboard teleop
cd ~/proj/IsaacLab
conda activate env_isaaclab
bash isaaclab.sh -p ~/proj/isaacsim-ros2-moveit/droid_sim/collect_data.py \
    --scene_id 1 --task "put the cube in the bowl" --num_episodes 5

# Or with ROS2 bridge (enables external data collection):
bash isaaclab.sh -p ~/proj/isaacsim-ros2-moveit/droid_sim/run_droid_ros2.py

# Terminal 2 (optional) -- External ROS2 data collector at 15 Hz
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 droid_sim/scripts/ros2_data_collector.py \
    --task "put the cube in the bowl" --output_dir ./droid_data
```

### Workflow 3: MCP Scene Scripting

```bash
# Terminal 1 -- Isaac Sim with MCP
./launch_isaacsim.sh

# Terminal 2 -- Run any MCP script
cd scripts && python setup_pick_place.py
# Or use interactively:
python -c "from mcp_client import IsaacMCP; mcp = IsaacMCP(); print(mcp.get_scene_info())"
```

## Critical Notes

- **Python conflict**: NEVER source `/opt/ros/humble/setup.bash` before launching Isaac Sim. Isaac Sim uses Python 3.11 internally; ROS2 Humble uses Python 3.10. Sourcing ROS2 first breaks Isaac Sim's Python environment.
- **Internal ROS2**: Isaac Sim uses its own rclpy via `setup_ros_env.sh`. External ROS2 nodes (MoveIt, RViz, data collector) use the system ROS2 and communicate via DDS.
- **RMW must match**: Set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` on both sides (Isaac Sim via setup_ros_env.sh, external nodes via export).
- **OmniGraph tick**: The OmniGraph needs a physx tick callback (`subscribe_physics_step_events`) to pulse `OnImpulseEvent` each frame. Without this, ROS2 topics will not publish.
- **MoveIt topics**: Isaac Sim publishes `/isaac_joint_states` and subscribes to `/isaac_joint_commands` (both `sensor_msgs/JointState`).
- **IsaacLab conda**: The DROID data collection scripts require the `env_isaaclab` conda environment and must be launched via `isaaclab.sh -p`.
- **Camera flag**: When using IsaacLab with cameras, always pass `--enable_cameras` (handled automatically in the scripts).

## Project Structure

```
isaac-sim-mcp-suite/
|-- CLAUDE.md                           # This file
|-- .gitignore
|-- LICENSE                             # MIT
|-- launch_isaacsim.sh                  # Launch Isaac Sim with MCP + ROS2 bridge
|-- launch_moveit.py                    # ROS2 launch: MoveIt2 + RViz + ros2_control
|-- rviz/
|   +-- moveit_isaac.rviz              # RViz config with MotionPlanning display
|-- scripts/
|   |-- mcp_client.py                  # Python TCP client for Isaac Sim MCP
|   |-- setup_moveit_scene.py          # Setup Franka + OmniGraph for MoveIt
|   +-- setup_pick_place.py            # Franka pick-and-place demo
|-- droid_sim/
|   |-- __init__.py
|   |-- collect_data.py                # IsaacLab data collection with keyboard teleop
|   |-- run_droid_ros2.py              # IsaacLab env + ROS2 bridge publishing
|   |-- test_fps.py                    # Minimal FPS benchmark
|   |-- envs/
|   |   |-- __init__.py
|   |   |-- droid_env.py               # DroidEnvCfg: scene, obs, actions, cameras
|   |   +-- franka_robotiq.py          # Franka + Robotiq 2F-85 ArticulationCfg
|   |-- recorder/
|   |   |-- __init__.py
|   |   +-- droid_recorder.py          # DROID HDF5 trajectory writer
|   |-- scripts/
|   |   |-- setup_droid_scene.py       # MCP-based DROID scene setup
|   |   |-- ros2_data_collector.py     # ROS2 node: sync + record at 15 Hz
|   |   |-- camera_viewer.py           # 3-camera OpenCV viewer
|   |   +-- replay_trajectory.py       # Replay recorded HDF5 in IsaacLab
|   |-- assets/                         # USD files (gitignored)
|   |   |-- franka_robotiq_2f_85_flattened.usd
|   |   |-- table.usd
|   |   +-- scene{1,2,3}.usd
|   +-- sim-evals/                     # Upstream sim-evals (submodule/gitignored)
|-- droid_data/                         # Recorded episodes (gitignored)
|   +-- episode_NNNN/
|       |-- trajectory.h5
|       +-- metadata.json
+-- docs/
    |-- mcp_commands.md                # MCP command reference
    |-- droid_data_format.md           # DROID HDF5 schema
    +-- architecture.md                # Detailed system architecture
```

## Dependencies

| Component | Location | Version |
|---|---|---|
| Isaac Sim | `~/isaac-sim/` | 5.1 |
| Isaac Sim MCP Extension | `~/Documents/isaac-sim-mcp/` | -- |
| IsaacLab | `~/proj/IsaacLab/` | 2.3.0 |
| ROS2 Humble | `/opt/ros/humble/` | system apt |
| MoveIt2 | `ros-humble-moveit` | apt |
| ros2_control | `ros-humble-ros2-control` | apt |
| topic_based_ros2_control | `ros-humble-topic-based-ros2-control` | apt |
| MCP Python lib | `pip install "mcp[cli]"` | >= 1.27 |
| h5py | pip | (for data recording/replay) |
| opencv-python | pip | (for camera viewer/image resize) |
| conda env: env_isaaclab | conda | (for IsaacLab scripts) |
