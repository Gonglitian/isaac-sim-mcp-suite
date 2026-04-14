"""
Setup and run Franka pick-and-place demo via MCP.

Creates Franka + blue cube scene, then runs the 7-phase pick-and-place controller.

Usage:
    python scripts/setup_pick_place.py
"""
from mcp_client import IsaacMCP
import sys

mcp = IsaacMCP()

if not mcp.is_connected():
    print("ERROR: Cannot connect to Isaac Sim MCP (localhost:8766)")
    sys.exit(1)

print("Connected to Isaac Sim.")

# Step 1: Setup scene
print("\n[Step 1] Creating pick-place scene...")
result = mcp.execute(r'''
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
import builtins

controller = FrankaPickPlace()
controller.setup_scene()
builtins._pick_place_controller = controller
print("Scene: Franka + blue cube + ground")
''')
print(f"  Result: {result.get('status')}")

# Step 2: Start pick-and-place
print("\n[Step 2] Starting pick-and-place...")
result = mcp.execute(r'''
import builtins
import omni.physx
import omni.timeline

controller = builtins._pick_place_controller

_step_count = [0]
def on_physics_step(dt):
    ctrl = builtins._pick_place_controller
    if ctrl is None:
        return
    if ctrl.is_done():
        if _step_count[0] >= 0:
            print("=== Pick-and-place DONE! ===")
            _step_count[0] = -1
        return
    try:
        ctrl.forward()
        _step_count[0] += 1
        if _step_count[0] % 50 == 0:
            print(f"Step {_step_count[0]}...")
    except Exception as e:
        print(f"Error: {e}")

builtins._physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
print("Simulation started - pick-and-place running!")
''')
print(f"  Result: {result.get('status')}")
print("\nWatch the Isaac Sim window!")
