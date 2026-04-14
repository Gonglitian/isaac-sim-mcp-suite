# isaac-sim-mcp-suite

A comprehensive robotics simulation toolkit for NVIDIA Isaac Sim that brings
together AI-assisted scene control (via MCP), ROS2-based motion planning
(MoveIt2), and DROID-compatible data collection in one repository.

```
+------------------------------------------------------------------+
|                      isaac-sim-mcp-suite                         |
|                                                                  |
|   +------------------+   +------------------+   +--------------+ |
|   |  MCP Extension   |   |  MoveIt2 Stack   |   | DROID Env    | |
|   |  (15+ commands)  |   |  (ROS2 + RViz)   |   | (IsaacLab)   | |
|   +--------+---------+   +--------+---------+   +------+-------+ |
|            |                      |                     |        |
|   +--------v---------+   +-------v----------+   +------v-------+ |
|   | Isaac Sim 5.1    <---+ ROS2 Bridge      +---> Data Collect | |
|   | (TCP :8766)      |   | (FastRTPS DDS)   |   | (HDF5 15Hz)  | |
|   +--------+---------+   +------------------+   +--------------+ |
+------------|-----------------------------------------------------+
             |
     +-------v--------+
     | AI Assistant    |
     | (Claude/Cursor) |
     +----------------+
```

**What you can do:**

- Control Isaac Sim scenes from Claude, Cursor, or any MCP-compatible AI
  assistant -- spawn robots, randomize environments, capture screenshots, and
  run arbitrary simulation scripts through natural language.
- Drag a Franka Panda end-effector interactively in RViz and watch the real
  motion execute in Isaac Sim via the MoveIt2 topic-based control bridge.
- Collect manipulation demonstrations in the DROID format (Franka + Robotiq
  2F-85 gripper + 3 cameras) and export them as HDF5 files ready for
  imitation-learning pipelines.


---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [MCP Commands Reference](#mcp-commands-reference)
7. [DROID Data Format](#droid-data-format)
8. [Project Structure](#project-structure)
9. [Important Notes](#important-notes)
10. [Acknowledgments](#acknowledgments)
11. [License](#license)


---

## Features

### MCP Extension (15+ commands)

- Full bidirectional control of Isaac Sim over TCP (port 8766)
- Scene introspection, object spawning/deletion, physics scenes, robot creation
- Domain randomization (object poses, lighting intensity/color, material colors)
- Save/load USD scenes, viewport and camera screenshots
- Direct robot joint control and state readback
- 3D asset generation from text or image (Beaver3D integration)
- USD asset search by text description

### MoveIt2 + RViz Integration

- MoveIt2 motion planning with OMPL, CHOMP, and Pilz planners
- Interactive end-effector dragging in RViz (plan and execute)
- Topic-based ROS2 control bridge to Isaac Sim
- Pre-configured RViz layout for Franka Panda

### DROID Data Collection

- Franka Panda + Robotiq 2F-85 gripper (matching real DROID hardware)
- 3 cameras: 2 external + 1 wrist-mounted (matching real DROID placement)
- Keyboard teleoperation (SE(3) delta + gripper toggle)
- DROID-compatible HDF5 output at 15 Hz
- Domain randomization across 3 table-top scenes
- ROS2-based async data collector (decoupled from sim loop)
- 3-camera side-by-side viewer
- Trajectory replay


---

## Architecture

The system is organized into three layers that can be used independently or
together:

```
Layer 1 -- AI Control (MCP)
+-------------------+          TCP :8766          +----------------------+
| AI Agent          | <-------------------------> | Isaac Sim Extension   |
| (Claude / Cursor) |     JSON commands/results   | (isaac.sim.mcp_ext)  |
+-------------------+                             +----------+-----------+
                                                             |
                                                    USD Stage + PhysX
                                                             |
Layer 2 -- Motion Planning (ROS2)                            |
+-------------------+       ROS2 Topics            +----------+-----------+
| MoveIt2 +  RViz   | <-------------------------> | Isaac ROS2 Bridge    |
| move_group        |  /isaac_joint_states         | (isaacsim.ros2)      |
| topic-based ctrl  |  /isaac_joint_commands       +----------------------+
+-------------------+

Layer 3 -- Data Collection (IsaacLab + ROS2)
+-------------------+       Internal rclpy         +----------------------+
| IsaacLab DROID Env| --------------------------> | ROS2 Topics           |
| (Franka+Robotiq)  |  /cam_ext1/rgb, /cam_ext2   | /isaac_joint_states  |
| Keyboard Teleop   |  /cam_wrist/rgb, /tf         | /clock               |
+-------------------+                             +----------+-----------+
                                                             |
                                                  +----------+-----------+
                                                  | ros2_data_collector  |
                                                  | 15 Hz -> HDF5        |
                                                  +----------------------+
```


---

## Prerequisites

| Component | Version | Notes |
|---|---|---|
| NVIDIA Isaac Sim | 5.1+ | GPU with driver 535+ recommended |
| ROS2 | Humble Hawksbill | `ros-humble-desktop` |
| IsaacLab | 2.3.0+ | Required only for DROID env |
| Python | 3.10+ | Isaac Sim ships its own 3.10 interpreter |
| MoveIt2 | Humble packages | `ros-humble-moveit`, `ros-humble-ros2-control`, `ros-humble-topic-based-ros2-control` |
| MCP CLI | latest | `pip install 'mcp[cli]'` |

Hardware: an NVIDIA GPU with at least 8 GB VRAM (RTX 3070 or better).


---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/isaac-sim-mcp-suite.git
cd isaac-sim-mcp-suite
```

### 2. Download DROID assets (for data collection only)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'owhan/DROID-sim-environments',
    repo_type='dataset',
    local_dir='droid/assets',
)
"
```

### 3. Set environment variables

```bash
# Point to your Isaac Sim installation
export ISAAC_SIM_DIR="$HOME/isaac-sim"

# ROS2 middleware -- MUST match on all terminals that communicate via ROS2
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 4. Install MCP server dependencies

```bash
pip install 'mcp[cli]'
```

### 5. Install MoveIt2 packages (for motion planning workflow)

```bash
sudo apt install ros-humble-moveit \
                 ros-humble-ros2-control \
                 ros-humble-topic-based-ros2-control \
                 ros-humble-moveit-resources-panda-moveit-config
```


---

## Quick Start

### Workflow A: MCP + MoveIt2 Interactive Control

Open three terminals.

**Terminal 1 -- Isaac Sim with MCP + ROS2 Bridge:**

```bash
# Do NOT source /opt/ros/humble/setup.bash here (see Important Notes)
./launch_isaacsim.sh
```

Wait for Isaac Sim to finish loading, then:

**Terminal 2 -- Setup scene via MCP:**

```bash
cd moveit/scripts
python setup_moveit_scene.py
```

This connects to Isaac Sim over TCP :8766, spawns a Franka Panda in a simple
room, and creates the ROS2 OmniGraph nodes for joint state publishing.

**Terminal 3 -- MoveIt2 + RViz:**

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch moveit/launch_moveit.py
```

In RViz, drag the interactive marker on the end-effector, then click
**Plan & Execute** to watch the robot move in Isaac Sim.

---

### Workflow B: DROID Data Collection via ROS2

**Terminal 1 -- IsaacLab DROID env with ROS2 publishing:**

```bash
cd ~/proj/IsaacLab
conda activate env_isaaclab
bash isaaclab.sh -p <path-to-repo>/droid/run_droid_ros2.py --scene_id 1
```

**Terminal 2 -- ROS2 data collector (15 Hz HDF5):**

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python droid/scripts/ros2_data_collector.py \
    --task "put the cube in the bowl" \
    --output_dir ./droid_data
```

Controls in the collector terminal: `s` start/stop, `n` save as success,
`f` save as failure, `q` quit.

---

### Workflow C: Direct DROID Data Collection (single process)

```bash
cd ~/proj/IsaacLab
conda activate env_isaaclab
bash isaaclab.sh -p <path-to-repo>/droid/collect_data.py \
    --scene_id 1 \
    --task "put the cube in the bowl" \
    --num_episodes 5 \
    --output_dir ./droid_data
```

Keyboard controls (focus the Isaac Sim viewport):

| Key | Action |
|---|---|
| W / S | Move X |
| A / D | Move Y |
| Q / E | Move Z |
| Z / X | Rotate roll |
| T / G | Rotate pitch |
| C / V | Rotate yaw |
| K | Toggle gripper |
| R | Reset episode |
| SPACE | Mark episode as success |
| ESC | Mark episode as failure |


---

## MCP Commands Reference

The MCP extension exposes these commands over TCP on `localhost:8766`. Each
command is also available as an MCP tool when using the `isaac_mcp` server
with Claude or Cursor.

| Command | Description | Key Parameters |
|---|---|---|
| `get_scene_info` | Ping the extension and retrieve asset root path | -- |
| `execute_script` | Run arbitrary Python code inside Isaac Sim | `code` (str) |
| `create_physics_scene` | Create a physics scene with floor and objects | `objects` (list), `floor` (bool), `gravity` (list) |
| `create_robot` | Spawn a robot from the Isaac Sim asset library | `robot_type` (franka/jetbot/carter/g1/go1), `position` |
| `get_all_poses` | Get world poses of all Xform prims under a path | `root_path` (default `/World`) |
| `get_robot_state` | Read joint positions, velocities, and EE pose | `robot_path` |
| `sim_control` | Play, pause, stop, step, reset, or query sim status | `action` (play/pause/stop/step/reset/status), `num_steps` |
| `screenshot` | Capture viewport or camera image to PNG | `camera_path`, `save_path`, `width`, `height` |
| `spawn_object` | Add a primitive (Cube/Sphere/Cylinder) or USD asset | `obj_type`, `name`, `position`, `scale`, `color`, `physics`, `usd_path` |
| `delete_object` | Remove a prim from the stage | `prim_path` |
| `randomize_scene` | Domain randomization: poses, lighting, colors | `randomize_objects`, `randomize_lighting`, `randomize_colors`, `object_pos_range` |
| `save_scene` | Export the current USD stage to a file | `file_path` |
| `load_scene` | Open a USD stage file | `file_path` |
| `set_robot_joints` | Directly set joint drive targets | `robot_path`, `joint_positions` (dict) |
| `transform` | Set position and scale of any prim | `prim_path`, `position`, `scale` |
| `generate_3d_from_text_or_image` | Generate a 3D model via Beaver3D | `text_prompt`, `image_url`, `position`, `scale` |
| `search_3d_usd_by_text` | Search USD libraries for a matching asset | `text_prompt`, `target_path` |

### MCP client configuration (Claude Desktop / Cursor)

Add the following to your MCP client config (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "isaac-sim": {
      "command": "python",
      "args": ["-m", "isaac_mcp.server"],
      "cwd": "<path-to-repo>/mcp_extension"
    }
  }
}
```


---

## DROID Data Format

Each episode is saved as an HDF5 file compatible with the
[DROID](https://arxiv.org/abs/2403.12945) dataset format:

```
episode_NNNN/trajectory.h5
|
+-- observation/
|   +-- robot_state/
|   |   +-- joint_positions      (T, 7)    float32
|   |   +-- joint_velocities     (T, 7)    float32
|   |   +-- gripper_position     (T,)      float32  [0=open, 1=closed]
|   |   +-- cartesian_position   (T, 6)    float32  [x,y,z,roll,pitch,yaw]
|   |
|   +-- camera_images/
|   |   +-- exterior_image_1_left  (T, 180, 320, 3)  uint8
|   |   +-- exterior_image_2_left  (T, 180, 320, 3)  uint8
|   |   +-- wrist_image_left       (T, 180, 320, 3)  uint8
|   |
|   +-- object_poses/
|       +-- <obj_name>/
|           +-- position           (T, 3)    float32
|           +-- orientation        (T, 4)    float32  [w,x,y,z]
|
+-- action/
|   +-- joint_position       (T, 7)    float32
|   +-- joint_velocity       (T, 7)    float32
|   +-- cartesian_position   (T, 6)    float32
|   +-- cartesian_velocity   (T, 6)    float32
|   +-- gripper_position     (T,)      float32
|   +-- gripper_velocity     (T,)      float32
|
+-- timestamps               (T,)      float64
```

Images are stored as raw uint8 arrays (180 x 320 x 3). A post-processing step
can convert them to MP4 for storage efficiency if needed.


---

## Project Structure

```
isaac-sim-mcp-suite/
|
+-- launch_isaacsim.sh                  # Launch Isaac Sim + MCP ext + ROS2 bridge
|
+-- mcp_extension/
|   +-- isaac_mcp/
|   |   +-- server.py                   # MCP server (FastMCP, TCP client to ext)
|   +-- isaac.sim.mcp_extension/
|   |   +-- config/extension.toml       # Extension metadata (port 8766)
|   |   +-- isaac_sim_mcp_extension/
|   |   |   +-- extension.py            # Isaac Sim extension (TCP server, 15+ handlers)
|   |   |   +-- gen3d.py                # Beaver3D integration
|   |   |   +-- usd.py                  # USD loader and search
|   |   +-- examples/                   # Example scripts (franka, g1, go1)
|   +-- LICENSE                         # MIT
|   +-- CHANGELOG.md
|
+-- moveit/
|   +-- launch_moveit.py                # ROS2 launch: move_group + RViz + controllers
|   +-- rviz/
|   |   +-- moveit_isaac.rviz           # Pre-configured RViz layout
|   +-- scripts/
|       +-- mcp_client.py               # Simple TCP client for the MCP extension
|       +-- setup_moveit_scene.py       # Create Franka scene + OmniGraph via MCP
|       +-- setup_pick_place.py         # Pick-and-place demo via MCP
|
+-- droid/
|   +-- envs/
|   |   +-- droid_env.py                # IsaacLab ManagerBasedRLEnv (scene, obs, actions)
|   |   +-- franka_robotiq.py           # Franka+Robotiq 2F-85 articulation config
|   +-- recorder/
|   |   +-- droid_recorder.py           # HDF5 trajectory recorder
|   +-- scripts/
|   |   +-- ros2_data_collector.py      # ROS2 async data collector (15 Hz)
|   |   +-- camera_viewer.py            # 3-camera side-by-side viewer
|   |   +-- replay_trajectory.py        # Replay recorded HDF5 trajectories
|   |   +-- setup_droid_scene.py        # Setup DROID scene in standalone Isaac Sim
|   +-- collect_data.py                 # Direct keyboard teleop + recording
|   +-- run_droid_ros2.py               # IsaacLab env + ROS2 bridge publishing
|   +-- assets/                         # Downloaded from HuggingFace (not in repo)
|
+-- docs/
+-- README.md
```


---

## Important Notes

**Python environment conflict.** Isaac Sim 5.1 bundles Python 3.10 and its own
`rclpy`. The system ROS2 Humble uses Python 3.11. If you `source
/opt/ros/humble/setup.bash` before launching Isaac Sim, the Python version
mismatch will cause import errors. Rule of thumb:

- Terminals that run Isaac Sim or IsaacLab: do NOT source the ROS2 setup script.
  Isaac Sim's `setup_ros_env.sh` handles the ROS2 environment internally.
- Terminals that run MoveIt2, data collectors, or other external ROS2 nodes:
  source `/opt/ros/humble/setup.bash` as usual.

**RMW implementation.** Both sides of every ROS2 connection must use the same
DDS middleware. Set this in every terminal:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

**IsaacLab uses internal rclpy.** The DROID env (`run_droid_ros2.py`) uses the
rclpy bundled inside IsaacLab. External ROS2 nodes (data collector, camera
viewer) use the system ROS2. They communicate over DDS, not through shared
Python objects.


---

## Acknowledgments

- The MCP extension is forked from
  [omni-mcp/isaac-sim-mcp](https://github.com/omni-mcp/isaac-sim-mcp) (MIT
  License).
- DROID simulation environments are adapted from
  [arhanjain/sim-evals](https://github.com/arhanjain/sim-evals).
- Robot and scene assets:
  [owhan/DROID-sim-environments](https://huggingface.co/datasets/owhan/DROID-sim-environments)
  on HuggingFace.
- DROID dataset and format: Khazatsky et al., "DROID: A Large-Scale In-the-Wild
  Robot Manipulation Dataset," 2024.
  [arXiv:2403.12945](https://arxiv.org/abs/2403.12945).


---

## License

This project is released under the [MIT License](mcp_extension/LICENSE).
