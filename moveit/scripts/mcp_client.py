"""
Simple TCP client for communicating with Isaac Sim MCP extension.

Usage:
    from scripts.mcp_client import IsaacMCP
    mcp = IsaacMCP()
    mcp.get_scene_info()
    mcp.execute("omni.kit.commands.execute('CreatePrim', prim_type='Cube')")
"""
import socket
import json
from typing import Any


class IsaacMCP:
    def __init__(self, host: str = "localhost", port: int = 8766, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _send(self, cmd_type: str, params: dict = None) -> dict:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.settimeout(self.timeout)
        cmd = json.dumps({"type": cmd_type, "params": params or {}})
        s.sendall(cmd.encode("utf-8"))
        data = s.recv(65536)
        result = json.loads(data.decode("utf-8"))
        s.close()
        return result

    def get_scene_info(self) -> dict:
        return self._send("get_scene_info")

    def execute(self, code: str) -> dict:
        return self._send("execute_script", {"code": code})

    def create_robot(self, robot_type: str = "franka", position: list = None) -> dict:
        return self._send("create_robot", {
            "robot_type": robot_type,
            "position": position or [0, 0, 0],
        })

    def create_physics_scene(self, objects: list = None, floor: bool = True) -> dict:
        return self._send("create_physics_scene", {
            "objects": objects or [],
            "floor": floor,
        })

    def transform(self, prim_path: str, position: list = None, scale: list = None) -> dict:
        return self._send("transform", {
            "prim_path": prim_path,
            "position": position or [0, 0, 0],
            "scale": scale or [1, 1, 1],
        })

    def is_connected(self) -> bool:
        try:
            result = self.get_scene_info()
            return result.get("status") == "success"
        except Exception:
            return False


if __name__ == "__main__":
    mcp = IsaacMCP()
    print("Connected:", mcp.is_connected())
    print(json.dumps(mcp.get_scene_info(), indent=2))
