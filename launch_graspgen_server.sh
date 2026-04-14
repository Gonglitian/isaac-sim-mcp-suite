#!/bin/bash
# Launch GraspGen ZMQ server (requires GPU + graspgen conda env)
#
# Prerequisites:
#   conda create -n graspgen python=3.10
#   conda activate graspgen
#   cd libs/GraspGen && pip install -e . && ./install_pointnet.sh
#   git clone https://huggingface.co/adithyamurali/GraspGenModels libs/GraspGenModels

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRASPGEN_DIR="${SCRIPT_DIR}/libs/GraspGen"
MODELS_DIR="${SCRIPT_DIR}/libs/GraspGenModels"
GRIPPER_CONFIG="${MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140.yml"
PORT="${1:-5556}"

if [ ! -f "$GRIPPER_CONFIG" ]; then
    echo "ERROR: Checkpoints not found at $MODELS_DIR"
    echo "Run: git clone https://huggingface.co/adithyamurali/GraspGenModels $MODELS_DIR"
    exit 1
fi

echo "=== GraspGen ZMQ Server ==="
echo "Gripper: robotiq_2f_140"
echo "Port: $PORT"
echo "=========================="

exec python "${GRASPGEN_DIR}/client-server/graspgen_server.py" \
    --gripper_config "$GRIPPER_CONFIG" \
    --port "$PORT"
