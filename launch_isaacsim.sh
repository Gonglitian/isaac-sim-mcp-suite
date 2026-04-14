#!/bin/bash
# Launch Isaac Sim with MCP extension + ROS2 Bridge
# NOTE: Do NOT source /opt/ros/humble/setup.bash before this script
#       (Python 3.10 vs 3.11 conflict — Isaac Sim uses internal rclpy)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaac-sim}"
MCP_EXT_DIR="${MCP_EXT_DIR:-$SCRIPT_DIR/mcp_extension}"

cd "$ISAAC_SIM_DIR"
source ./setup_ros_env.sh

echo "=== Environment ==="
echo "ROS_DISTRO=$ROS_DISTRO"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "Isaac Sim: $ISAAC_SIM_DIR"
echo "MCP Extension: $MCP_EXT_DIR"
echo "==================="

exec ./isaac-sim.sh \
  --ext-folder "$MCP_EXT_DIR" \
  --enable isaac.sim.mcp_extension \
  --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
