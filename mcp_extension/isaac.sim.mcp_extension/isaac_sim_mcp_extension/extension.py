"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Extension module for Isaac Sim MCP."""

import asyncio
import carb
# import omni.ext
# import omni.ui as ui
import omni.usd
import threading
import time
import socket
import json
import traceback

import gc
from pxr import Usd, UsdGeom, Sdf, Gf

import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World
# Import Beaver3d and USDLoader
from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader
from isaac_sim_mcp_extension.usd import USDSearch3d
import requests

# Extension Methods required by Omniverse Kit
# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MCPExtension(omni.ext.IExt):
    def __init__(self) -> None:
        """Initialize the extension."""
        super().__init__()
        self.ext_id = None
        self.running = False
        self.host = None
        self.port = None
        self.socket = None
        self.server_thread = None
        self._usd_context = None
        self._physx_interface = None
        self._timeline = None
        self._window = None
        self._status_label = None
        self._server_thread = None
        self._models = None
        self._settings = carb.settings.get_settings()
        self._image_url_cache = {} # cache for image url
        self._text_prompt_cache = {} # cache for text prompt
        

    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        print("trigger  on_startup for: ", ext_id)
        print("settings: ", self._settings.get("/exts/omni.kit.pipapi"))
        self.port = self._settings.get("/exts/isaac.sim.mcp/server, port") or 8766
        self.host = self._settings.get("/exts/isaac.sim.mcp/server.host") or "localhost"
        if not hasattr(self, 'running'):
            self.running = False

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        # omni.kit.commands.execute("CreatePrim", prim_type="Sphere")

        # print("sphere created")
        # result = self.execute_script('omni.kit.commands.execute("CreatePrim", prim_type="Cube")')
        # print("script executed", result)  
        self._start()
        # result = self.execute_script('omni.kit.commands.execute("CreatePrim", prim_type="Cube")')
        # print("script executed", result)  
    
    def on_shutdown(self):
        print("trigger  on_shutdown for: ", self.ext_id)
        self._models = {}
        gc.collect()
        self._stop()
    
    def _start(self):
        if self.running:
            print("Server is already running")
            return
            
        self.running = True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"Isaac Sim MCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()
            
    def _stop(self):
        self.running = False
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        
        print("Isaac Sim MCP server stopped")

    def _server_loop(self):
        """Main server loop in a separate thread"""
        print("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping
        if not hasattr(self, 'running'):
            self.running = False

        while self.running:
            try:
                # Accept new connection
                try:
                    client, address = self.socket.accept()
                    print(f"Connected to client: {address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(0.5)
        
        print("Server thread stopped")
    
    def _handle_client(self, client):
        """Handle connected client"""
        print("Client handler started")
        client.settimeout(None)  # No timeout
        buffer = b''
        
        try:
            while self.running:
                # Receive data
                try:
                    data = client.recv(16384)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    buffer += data
                    try:
                        # Try to parse command
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''
                        
                        # Execute command in Isaac Sim's main thread
                        async def execute_wrapper():
                            try:
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                print("response_json: ", response_json)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print("Failed to send response - client disconnected")
                            except Exception as e:
                                print(f"Error executing command: {str(e)}")
                                traceback.print_exc()
                                try:
                                    error_response = {
                                        "status": "error",
                                        "message": str(e)
                                    }
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None
                        # import omni.kit.commands
                        # import omni.kit.async
                        from omni.kit.async_engine import run_coroutine
                        task = run_coroutine(execute_wrapper())
                        # import asyncio
                        # asyncio.ensure_future(execute_wrapper())
                        #time.sleep(30)
                        
    
                        # 
                        # omni.kit.async.get_event_loop().create_task(create_sphere_async())
                        # TODO:Schedule execution in main thread
                        # bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                        # omni.kit.app.get_app().post_to_main_thread(execute_wrapper())
                        # carb.apputils.get_app().get_update_event_loop().post(execute_wrapper)

                        # from omni.kit.async_engine import run_coroutine
                        # run_coroutine(execute_wrapper())
                        # omni.kit.app.get_app().get_update_event_stream().push(0, 0, {"fn": execute_wrapper})
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        pass
                except Exception as e:
                    print(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            print(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            print("Client handler stopped")

    # TODO: This is a temporary function to execute commands in the main thread
    def execute_command(self, command):
        """Execute a command in the main thread"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            # Ensure USD context for object operations
            if cmd_type in ["create_object", "modify_object", "delete_object"]:
                self._usd_context = omni.usd.get_context()
            return self._execute_command_internal(command)
                
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        #todo: add a handler for extend simulation method if necessary
        handlers = {
            "execute_script": self.execute_script,
            "get_scene_info": self.get_scene_info,
            "omini_kit_command": self.omini_kit_command,
            "create_physics_scene": self.create_physics_scene,
            "create_robot": self.create_robot,
            "generate_3d_from_text_or_image": self.generate_3d_from_text_or_image,
            "transform": self.transform,
            "search_3d_usd_by_text": self.search_3d_usd_by_text,
            # === New commands ===
            "get_all_poses": self.get_all_poses,
            "get_robot_state": self.get_robot_state,
            "sim_control": self.sim_control,
            "screenshot": self.screenshot,
            "spawn_object": self.spawn_object,
            "delete_object": self.delete_object,
            "randomize_scene": self.randomize_scene,
            "save_scene": self.save_scene,
            "load_scene": self.load_scene,
            "set_robot_joints": self.set_robot_joints,
        }
        
        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete: /n", result)
                # return result
                if result and result.get("status") == "success":   
                    return {"status": "success", "result": result}
                else:
                    return {"status": "error", "message": result.get("message", "Unknown error")}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
        

    

    def execute_script(self, code: str) :
        """Execute a Python script within the Isaac Sim context.
        
        Args:
            code: The Python script to execute.
            
        Returns:
            Dictionary with execution result.
        """
        try:
            # Create a local namespace
            local_ns = {}
            
            # Add frequently used modules to the namespace
            local_ns["omni"] = omni
            local_ns["carb"] = carb
            local_ns["Usd"] = Usd
            local_ns["UsdGeom"] = UsdGeom
            local_ns["Sdf"] = Sdf
            local_ns["Gf"] = Gf
            # code = script["code"]
            
            # Execute the script
            exec(code,  local_ns)
            
            # Get the result if any
            # result = local_ns.get("result", None)
            result = None
            
            
            return {
                "status": "success",
                "message": "Script executed successfully",
                "result": result
            }
        except Exception as e:
            carb.log_error(f"Error executing script: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
    def get_scene_info(self):
        self._stage = omni.usd.get_context().get_stage()
        assert self._stage is not None
        stage_path = self._stage.GetRootLayer().realPath
        assets_root_path = get_assets_root_path()
        return {"status": "success", "message": "pong", "assets_root_path": assets_root_path}
        
    def omini_kit_command(self,  command: str, prim_type: str) -> Dict[str, Any]:
        omni.kit.commands.execute(command, prim_type=prim_type)
        print("command executed")
        return {"status": "success", "message": "command executed"}
    
    def create_robot(self, robot_type: str = "g1", position: List[float] = [0, 0, 0]):
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
        from omni.isaac.nucleus import get_assets_root_path
        

        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        print("position: ", position)
        
        if robot_type.lower() == "franka":
            asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
            add_reference_to_stage(asset_path, "/Franka")
            robot_prim = XFormPrim(prim_path="/Franka")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
        elif robot_type.lower() == "jetbot":
            asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
            add_reference_to_stage(asset_path, "/Jetbot")
            robot_prim = XFormPrim(prim_path="/Jetbot")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
        elif robot_type.lower() == "carter":
            asset_path = assets_root_path + "/Isaac/Robots/Carter/carter.usd"
            add_reference_to_stage(asset_path, "/Carter")
            robot_prim = XFormPrim(prim_path="/Carter")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
        elif robot_type.lower() == "g1":
            asset_path = assets_root_path + "/Isaac/Robots/Unitree/G1/g1.usd"
            add_reference_to_stage(asset_path, "/G1")
            robot_prim = XFormPrim(prim_path="/G1")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
        elif robot_type.lower() == "go1":
            asset_path = assets_root_path + "/Isaac/Robots/Unitree/Go1/go1.usd"
            add_reference_to_stage(asset_path, "/Go1")
            robot_prim = XFormPrim(prim_path="/Go1")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
        else:
            # Default to Franka if unknown robot type
            asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
            add_reference_to_stage(asset_path, "/Franka")
            robot_prim = XFormPrim(prim_path="/Franka")
            robot_prim.set_world_pose(position=np.array(position))
            return {"status": "success", "message": f"{robot_type} robot created"}
    
    def create_physics_scene(
            self,
            objects: List[Dict[str, Any]] = [],
            floor: bool = True,
            gravity: List[float] = (0.0, -9.81, 0.0),
            scene_name: str = "None"
        ) -> Dict[str, Any]:
            """Create a physics scene with multiple objects."""
            try:
                # Set default values
                gravity = gravity or [0, -9.81, 0]
                scene_name = scene_name or "physics_scene"
                
                
                # Create a new stage
                #omni.kit.commands.execute("CreateNewStage")
                
                
                stage = omni.usd.get_context().get_stage()
                print("stage: ", stage)
                
                # print("start to create new sphere")
                # # import omni.kit.commands
                # omni.kit.commands.execute("CreatePrim", prim_type="Sphere")
                # print("create sphere successfully")
                
                # Set up the physics scene
                scene_path = "/World/PhysicsScene"
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=scene_path,
                    prim_type="PhysicsScene",
                    
                )
                #attributes={"physxScene:enabled": True , "physxScene:gravity": gravity},
                

                # Initialize simulation context with physics
                # simulation_context = SimulationContext()
                # my_world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        
                # # Make sure the world is playing before initializing the robot
                # if not my_world.is_playing():
                #     my_world.play()
                #     # Wait a few frames for physics to stabilize
                # for _ in range(1000):
                #     my_world.step_async()
                # my_world.initialize_physics()

                # print("created physics scene: ", scene_path)
                
                # Create the World prim as a Xform
                world_path = "/World"
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=world_path,
                    prim_type="Xform",
                )
                print("create world: ", world_path)
                # Create a ground plane if requested
                if floor:
                    floor_path = "/World/ground"
                    omni.kit.commands.execute(
                        "CreatePrim",
                        prim_path=floor_path,
                        prim_type="Plane",
                        attributes={"size": 100.0}  # Large ground plane
                    )
                    
                    # Add physics properties to the ground
                    # omni.kit.commands.execute(
                    #     "CreatePhysics",
                    #     prim_path=floor_path,
                    #     physics_type="collider",
                    #     attributes={
                    #         "static": True,
                    #         "collision_enabled": True
                    #     }
                    # )
                # objects = [
                # {"path": "/World/Cube", "type": "Cube", "size": 20, "position": (0, 100, 0), "rotation": [1, 2, 3, 0], "scale": [1, 1, 1], "color": [0.5, 0.5, 0.5, 1.0], "physics_enabled": True, "mass": 1.0, "is_kinematic": False},
                # {"path": "/World/Sphere", "type": "Sphere", "radius": 5, "position": (5, 200, 0)},
                # {"path": "/World/Cone", "type": "Cone", "height": 8, "radius": 3, "position": (-5, 150, 0)}
                # ]
                print("start create objects: ", objects)
                objects_created = 0
                # Create each object
                for i, obj in enumerate(objects):
                    obj_name = obj.get("name", f"object_{i}")
                    obj_type = obj.get("type", "Cube")
                    obj_position = obj.get("position", [0, 0, 0])
                    obj_rotation = obj.get("rotation", [1, 0, 0, 0])  # Default is no rotation (identity quaternion)
                    obj_scale = obj.get("scale", [1, 1, 1])
                    obj_color = obj.get("color", [0.5, 0.5, 0.5, 1.0])
                    obj_physics = obj.get("physics_enabled", True)
                    obj_mass = obj.get("mass", 1.0)
                    obj_kinematic = obj.get("is_kinematic", False)
                    
                    # Create the object
                    obj_path = obj.get("path", f"/World/{obj_name}")
                    print("obj_path: ", obj_path)
                    if stage.GetPrimAtPath(obj_path):
                        print("obj_path already exists and skip creating")
                        continue
                    
                    # Create the primitive based on type
                    if obj_type in ["Cube", "Sphere", "Cylinder", "Cone", "Plane"]:
                        omni.kit.commands.execute(
                            "CreatePrim",
                            prim_path=obj_path,
                            prim_type=obj_type,
                            attributes={
                                "size": obj.get("size", 100.0), 
                                "position": obj_position, 
                                "rotation": obj_rotation, 
                                "scale": obj_scale, 
                                "color": obj_color, 
                                "physics_enabled": obj_physics,
                                "mass": obj_mass,
                                "is_kinematic": obj_kinematic} if obj_type in ["Cube", "Sphere","Plane"] else {},
                        )
                        print(f"Created {obj_type} at {obj_path}")
                    else:
                        return {"status": "error", "message": f"Invalid object type: {obj_type}"}
                    
                    # Set the transform
                    omni.kit.commands.execute(
                        "TransformPrimSRT",
                        path=obj_path,
                        new_translation=obj_position,
                        new_rotation_euler=[0, 0, 0],  # We'll set the quaternion separately
                        new_scale=obj_scale,
                    )
                    print(f"Created TransformPrimSRT at {obj_position}")
                    # Set rotation as quaternion
                    xform = UsdGeom.Xformable(stage.GetPrimAtPath(obj_path))
                    if xform and obj_rotation != [1, 0, 0, 0]:
                        quat = Gf.Quatf(obj_rotation[0], obj_rotation[1], obj_rotation[2], obj_rotation[3])
                        xform_op = xform.AddRotateOp()
                        xform_op.Set(quat)
                    
                    # Add physics properties if enabled
                    if obj_physics:
                        omni.kit.commands.execute(
                            "CreatePhysics",
                            prim_path=obj_path,
                            physics_type="rigid_body" if not obj_kinematic else "kinematic_body",
                            attributes={
                                "mass": obj_mass,
                                "collision_enabled": True,
                                "kinematic": obj_kinematic
                            }
                        )
                    print(f"Created Physics at {obj_path}")
                    # Set the color
                    if obj_color:
                        material_path = f"{obj_path}/material"
                        omni.kit.commands.execute(
                            "CreatePrim",
                            prim_path=material_path,
                            prim_type="Material",
                            attributes={
                                "diffuseColor": obj_color[:3],
                                "opacity": obj_color[3] if len(obj_color) > 3 else 1.0
                            }
                        )
                        print(f"Created Material at {material_path}")
                        # Bind the material to the object
                        omni.kit.commands.execute(
                            "BindMaterial",
                            material_path=material_path,
                            prim_path=obj_path
                        )

                        print(f"Bound Material to {obj_path}")
                        # increment the number of objects created
                        objects_created += 1
                return {
                    "status": "success",
                    "message": f"Created physics scene with {objects_created} objects",
                    "result": scene_name
                }
                
            except Exception as e:
                import traceback
                return {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
   
    def generate_3d_from_text_or_image(self, text_prompt=None, image_url=None, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Generate a 3D model from text or image, load it into the scene and transform it.
        
        Args:
            text_prompt (str, optional): Text prompt for 3D generation
            image_url (str, optional): URL of image for 3D generation
            position (tuple, optional): Position to place the model
            scale (tuple, optional): Scale of the model
            
        Returns:
            dict: Dictionary with the task_id and prim_path
        """
        try:
            # Initialize Beaver3d
            beaver = Beaver3d()
            
            # Determine generation method based on inputs
            # if image_url and text_prompt:
            #     # Generate 3D from image with text prompt as options
            #     task_id = beaver.generate_3d_from_image(image_url, text_prompt)
            #     print(f"3D model generation from image with text options started with task ID: {task_id}")
            # Check if we have cached task IDs for this input
            if not hasattr(self, '_image_url_cache'):
                self._image_url_cache = {}  # Cache for image URL to task_id mapping
            
            if not hasattr(self, '_text_prompt_cache'):
                self._text_prompt_cache = {}  # Cache for text prompt to task_id mapping
            
            # Check if we can retrieve task_id from cache
            task_id = None
            if image_url and image_url in self._image_url_cache:
                task_id = self._image_url_cache[image_url]
                print(f"Using cached task ID: {task_id} for image URL: {image_url}")
            elif text_prompt and text_prompt in self._text_prompt_cache:
                task_id = self._text_prompt_cache[text_prompt]
                print(f"Using cached task ID: {task_id} for text prompt: {text_prompt}")

            if task_id: #cache hit
                print(f"Using cached model ID: {task_id}")
            elif image_url:
                # Generate 3D from image only
                task_id = beaver.generate_3d_from_image(image_url)
                print(f"3D model generation from image started with task ID: {task_id}")
            elif text_prompt:
                # Generate 3D from text
                task_id = beaver.generate_3d_from_text(text_prompt)
                print(f"3D model generation from text started with task ID: {task_id}")
            else:
                return {
                    "status": "error",
                    "message": "Either text_prompt or image_url must be provided"
                }
            
            # Monitor the task and download the result
            # result_path = beaver.monitor_task_status(task_id)
            # task = asyncio.create_task(
                # beaver.monitor_task_status_async(
                    # task_id, on_complete_callback=load_model_into_scene))
            #await task
            def load_model_into_scene(task_id, status, result_path):
                print(f"{task_id} is {status}, 3D model  downloaded to: {result_path}")
                # Only cache the task_id after successful download
                if image_url and image_url not in self._image_url_cache:
                    self._image_url_cache[image_url] = task_id
                elif text_prompt and text_prompt not in self._text_prompt_cache:
                    self._text_prompt_cache[text_prompt] = task_id
                # Load the model into the scene
                loader = USDLoader()
                prim_path = loader.load_usd_model(task_id=task_id)
                
                # Load texture and create material
                try:
                    texture_path, material = loader.load_texture_and_create_material(task_id=task_id)
                    
                    # Bind texture to model
                    loader.bind_texture_to_model()
                except Exception as e:
                    print(f"Warning: Texture loading failed, continuing without texture: {str(e)}")
                
                # Transform the model
                loader.transform(position=position, scale=scale)
            
                return {
                    "status": "success",
                    "task_id": task_id,
                    "prim_path": prim_path
                }
            
            from omni.kit.async_engine import run_coroutine
            task = run_coroutine(beaver.monitor_task_status_async(
                task_id, on_complete_callback=load_model_into_scene))
            
            return {
                    "status": "success",
                    "task_id": task_id,
                    "message": f"3D model generation started with task ID: {task_id}"
            }
            
            
            
        except Exception as e:
            print(f"Error generating 3D model: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }
    
    def search_3d_usd_by_text(self, text_prompt:str, target_path:str, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Search a USD assets in USD Search service, load it into the scene and transform it.
        
        Args:
            text_prompt (str, optional): Text prompt for 3D generation
            target_path (str, ): target path in current scene stage
            position (tuple, optional): Position to place the model
            scale (tuple, optional): Scale of the model
            
        Returns:
            dict: Dictionary with prim_path
        """
        try:
            if text_prompt:
                print(f"3D model generation from text: {text_prompt}")
            else:
                return {
                    "status": "error",
                    "message": "text_prompt must be provided"
                }
            
            searcher3d = USDSearch3d()
            url = searcher3d.search( text_prompt )

            loader = USDLoader()
            prim_path = loader.load_usd_from_url( url, target_path )
            print(f"loaded url {url} to scene, prim path is: {prim_path}")
            # TODO: transform the model, need to fix the transform function for loaded USD
            # loader.transform(prim=prim_path, position=position, scale=scale)
            
            return {
                    "status": "success",
                    "prim_path": prim_path,
                    "message": f"3D model searching with prompt: {text_prompt}, return url: {url}, prim path in current scene: {prim_path}"
            }
        except Exception as e:
            print(f"Error searching 3D model: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }
    
    # ================================================================
    # P0: Data collection essentials
    # ================================================================

    def get_all_poses(self, root_path: str = "/World"):
        """Get poses of all Xform/rigid prims under root_path."""
        try:
            stage = omni.usd.get_context().get_stage()
            results = {}
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(root_path):
                    continue
                xformable = UsdGeom.Xformable(prim)
                if not xformable:
                    continue
                try:
                    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = xform.ExtractTranslation()
                    rot = xform.ExtractRotationQuat()
                    results[path] = {
                        "position": [pos[0], pos[1], pos[2]],
                        "orientation": [rot.GetReal(), rot.GetImaginary()[0],
                                        rot.GetImaginary()[1], rot.GetImaginary()[2]],
                    }
                except Exception:
                    continue
            return {"status": "success", "poses": results, "count": len(results)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_robot_state(self, robot_path: str = "/World/envs/env_0/robot"):
        """Get robot joint positions, velocities, EE pose, gripper state."""
        try:
            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_path)
            if not robot_prim.IsValid():
                return {"status": "error", "message": f"Robot not found at {robot_path}"}

            from pxr import UsdPhysics, PhysxSchema
            import omni.isaac.core.utils.prims as prim_utils

            # Collect joint states
            joints = {}
            for prim in Usd.PrimRange(robot_prim):
                joint = UsdPhysics.RevoluteJoint(prim) or UsdPhysics.PrismaticJoint(prim)
                if not joint:
                    continue
                name = prim.GetName()
                # Try to read drive target/position
                drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Get(prim, "linear")
                if drive:
                    target = drive.GetTargetPositionAttr().Get()
                    joints[name] = {"target": target}

            # EE pose (last link)
            ee_paths = [
                f"{robot_path}/panda_link8",
                f"{robot_path}/Gripper/Robotiq_2F_85/base_link",
            ]
            ee_pose = None
            for ee_path in ee_paths:
                ee_prim = stage.GetPrimAtPath(ee_path)
                if ee_prim.IsValid():
                    xform = UsdGeom.Xformable(ee_prim)
                    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = mat.ExtractTranslation()
                    rot = mat.ExtractRotationQuat()
                    ee_pose = {
                        "position": [pos[0], pos[1], pos[2]],
                        "orientation": [rot.GetReal(), rot.GetImaginary()[0],
                                        rot.GetImaginary()[1], rot.GetImaginary()[2]],
                        "frame": ee_path,
                    }
                    break

            return {
                "status": "success",
                "joints": joints,
                "ee_pose": ee_pose,
                "robot_path": robot_path,
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def sim_control(self, action: str = "status", num_steps: int = 1):
        """Control simulation: play, pause, stop, step, reset, status."""
        try:
            timeline = omni.timeline.get_timeline_interface()
            if action == "play":
                timeline.play()
                return {"status": "success", "message": "Playing"}
            elif action == "pause":
                timeline.pause()
                return {"status": "success", "message": "Paused"}
            elif action == "stop":
                timeline.stop()
                return {"status": "success", "message": "Stopped"}
            elif action == "step":
                for _ in range(num_steps):
                    omni.kit.app.get_app().update()
                return {"status": "success", "message": f"Stepped {num_steps} frames"}
            elif action == "reset":
                timeline.stop()
                timeline.play()
                return {"status": "success", "message": "Reset (stop+play)"}
            elif action == "status":
                return {
                    "status": "success",
                    "playing": timeline.is_playing(),
                    "stopped": timeline.is_stopped(),
                    "current_time": timeline.get_current_time(),
                }
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def screenshot(self, camera_path: str = "viewport", save_path: str = "/tmp/mcp_screenshot.png",
                   width: int = 640, height: int = 480):
        """Capture screenshot from viewport or any camera prim.

        Args:
            camera_path: "viewport" for main viewport, or a camera prim path
                         e.g. "/World/envs/env_0/external_cam_1"
            save_path: Where to save the PNG file.
            width: Image width (only for camera prims, viewport uses its own size).
            height: Image height (only for camera prims).
        """
        try:
            import os
            os.makedirs(os.path.dirname(save_path) or "/tmp", exist_ok=True)

            if camera_path == "viewport" or camera_path == "":
                # Viewport screenshot
                import omni.kit.viewport.utility as vp_utils
                from omni.kit.viewport.utility import capture_viewport_to_file
                viewport = vp_utils.get_active_viewport()
                if viewport is None:
                    return {"status": "error", "message": "No active viewport"}
                capture_viewport_to_file(viewport, save_path)
                return {
                    "status": "success",
                    "save_path": save_path,
                    "source": "viewport",
                    "message": f"Viewport screenshot saved to {save_path}",
                }
            else:
                # Camera screenshot: temporarily switch viewport camera, capture, switch back
                import omni.kit.viewport.utility as vp_utils
                from omni.kit.viewport.utility import capture_viewport_to_file
                viewport = vp_utils.get_active_viewport()
                if viewport is None:
                    return {"status": "error", "message": "No active viewport"}

                # Resolve camera prim path
                stage = omni.usd.get_context().get_stage()
                cam_prim = None
                # Search for matching camera prim
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    if camera_path in path and prim.GetTypeName() == "Camera":
                        cam_prim = path
                        break
                if cam_prim is None:
                    return {"status": "error", "message": f"Camera prim not found matching: {camera_path}"}

                # Save current camera, switch, capture, restore
                original_cam = viewport.get_active_camera()
                viewport.set_active_camera(cam_prim)
                # Render a few frames with new camera
                for _ in range(3):
                    omni.kit.app.get_app().update()
                capture_viewport_to_file(viewport, save_path)
                # Wait for file write
                for _ in range(3):
                    omni.kit.app.get_app().update()
                # Restore original camera
                viewport.set_active_camera(str(original_cam))

                return {
                    "status": "success",
                    "save_path": save_path,
                    "source": cam_prim,
                    "message": f"Camera screenshot saved to {save_path}",
                }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    # ================================================================
    # P1: Scene diversity / domain randomization
    # ================================================================

    def spawn_object(self, obj_type: str = "Cube", name: str = "object",
                     position: list = [0, 0, 0.5], scale: list = [0.05, 0.05, 0.05],
                     color: list = [0.5, 0.5, 0.5], physics: bool = True,
                     usd_path: str = ""):
        """Spawn a primitive or USD object into the scene."""
        try:
            stage = omni.usd.get_context().get_stage()
            prim_path = f"/World/envs/env_0/{name}"

            if stage.GetPrimAtPath(prim_path).IsValid():
                return {"status": "error", "message": f"Prim already exists: {prim_path}"}

            if usd_path:
                # Load from USD file
                from isaacsim.core.utils.stage import add_reference_to_stage
                add_reference_to_stage(usd_path, prim_path)
            else:
                # Create primitive
                prim = stage.DefinePrim(prim_path, obj_type)
                if obj_type == "Cube":
                    UsdGeom.Cube(prim).GetSizeAttr().Set(1.0)
                elif obj_type == "Sphere":
                    UsdGeom.Sphere(prim).GetRadiusAttr().Set(0.5)
                elif obj_type == "Cylinder":
                    UsdGeom.Cylinder(prim).GetRadiusAttr().Set(0.5)
                    UsdGeom.Cylinder(prim).GetHeightAttr().Set(1.0)

            prim = stage.GetPrimAtPath(prim_path)

            # Transform
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            xform.AddScaleOp().Set(Gf.Vec3d(*scale))

            # Physics
            if physics:
                from pxr import UsdPhysics
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.CollisionAPI.Apply(prim)

            # Color
            if color and not usd_path:
                mat_path = f"{prim_path}/material"
                mat = omni.usd.get_context().get_stage().DefinePrim(mat_path, "Material")
                from pxr import UsdShade
                material = UsdShade.Material.Define(stage, mat_path)
                shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*color))
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(prim).Bind(material)

            return {"status": "success", "prim_path": prim_path,
                    "message": f"Spawned {obj_type} at {position}"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def delete_object(self, prim_path: str):
        """Delete a prim from the scene."""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return {"status": "error", "message": f"Prim not found: {prim_path}"}
            stage.RemovePrim(prim_path)
            return {"status": "success", "message": f"Deleted {prim_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def randomize_scene(self, randomize_objects: bool = True,
                        randomize_lighting: bool = True,
                        randomize_colors: bool = False,
                        object_pos_range: list = [[-0.3, -0.3, 0], [0.3, 0.3, 0]],
                        root_path: str = "/World/envs/env_0/scene"):
        """Domain randomization: randomize object poses, lighting, colors."""
        try:
            import random
            stage = omni.usd.get_context().get_stage()
            changes = []

            if randomize_objects:
                root_prim = stage.GetPrimAtPath(root_path)
                if root_prim.IsValid():
                    for child in root_prim.GetChildren():
                        # Skip non-geometric prims
                        if not UsdGeom.Xformable(child):
                            continue
                        name = child.GetName()
                        if name in ("table", "DomeLight", "PhysicsMaterial"):
                            continue
                        # Randomize position
                        xform = UsdGeom.Xformable(child)
                        lo, hi = object_pos_range
                        new_pos = Gf.Vec3d(
                            random.uniform(lo[0], hi[0]),
                            random.uniform(lo[1], hi[1]),
                            random.uniform(lo[2], hi[2]) if len(lo) > 2 else 0.5,
                        )
                        # Random rotation around Z
                        angle = random.uniform(0, 360)
                        ops = xform.GetOrderedXformOps()
                        if ops:
                            ops[0].Set(new_pos)
                        else:
                            xform.ClearXformOpOrder()
                            xform.AddTranslateOp().Set(new_pos)
                        changes.append(f"{name} -> pos={[round(x,3) for x in new_pos]}")

            if randomize_lighting:
                # Find and randomize lights
                for prim in stage.Traverse():
                    if "Light" in prim.GetTypeName():
                        try:
                            light_prim = prim
                            # Randomize intensity
                            intensity_attr = light_prim.GetAttribute("inputs:intensity")
                            if intensity_attr:
                                new_intensity = random.uniform(2000, 8000)
                                intensity_attr.Set(new_intensity)
                                changes.append(f"light intensity -> {new_intensity:.0f}")
                            # Randomize color temperature
                            color_attr = light_prim.GetAttribute("inputs:color")
                            if color_attr:
                                r = random.uniform(0.8, 1.0)
                                g = random.uniform(0.8, 1.0)
                                b = random.uniform(0.8, 1.0)
                                color_attr.Set(Gf.Vec3f(r, g, b))
                                changes.append(f"light color -> ({r:.2f},{g:.2f},{b:.2f})")
                        except Exception:
                            continue

            if randomize_colors:
                root_prim = stage.GetPrimAtPath(root_path)
                if root_prim.IsValid():
                    for child in root_prim.GetChildren():
                        name = child.GetName()
                        if name in ("table", "DomeLight", "PhysicsMaterial"):
                            continue
                        # Create random color material
                        mat_path = f"{child.GetPath()}/rand_mat"
                        if stage.GetPrimAtPath(mat_path).IsValid():
                            stage.RemovePrim(mat_path)
                        from pxr import UsdShade
                        material = UsdShade.Material.Define(stage, mat_path)
                        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
                        shader.CreateIdAttr("UsdPreviewSurface")
                        r, g, b = random.random(), random.random(), random.random()
                        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f(r, g, b))
                        material.CreateSurfaceOutput().ConnectToSource(
                            shader.ConnectableAPI(), "surface")
                        UsdShade.MaterialBindingAPI(child).Bind(material)
                        changes.append(f"{name} color -> ({r:.2f},{g:.2f},{b:.2f})")

            return {"status": "success", "changes": changes, "count": len(changes)}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    # ================================================================
    # P2: Save/load, direct joint control
    # ================================================================

    def save_scene(self, file_path: str = ""):
        """Save current USD stage to file."""
        try:
            stage = omni.usd.get_context().get_stage()
            if not file_path:
                file_path = stage.GetRootLayer().realPath
                if not file_path:
                    file_path = "/tmp/saved_scene.usd"
            stage.GetRootLayer().Export(file_path)
            return {"status": "success", "file_path": file_path, "message": f"Scene saved to {file_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_scene(self, file_path: str = ""):
        """Load a USD scene file."""
        try:
            if not file_path:
                return {"status": "error", "message": "file_path required"}
            omni.usd.get_context().open_stage(file_path)
            return {"status": "success", "file_path": file_path, "message": f"Scene loaded from {file_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def set_robot_joints(self, robot_path: str = "/World/envs/env_0/robot",
                         joint_positions: dict = {}):
        """Set robot joint positions directly. joint_positions: {joint_name: value_in_radians}."""
        try:
            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_path)
            if not robot_prim.IsValid():
                return {"status": "error", "message": f"Robot not found: {robot_path}"}

            from pxr import UsdPhysics
            set_joints = []
            for prim in Usd.PrimRange(robot_prim):
                name = prim.GetName()
                if name in joint_positions:
                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                    if not drive:
                        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                    if drive:
                        drive.GetTargetPositionAttr().Set(float(joint_positions[name]))
                        set_joints.append(name)

            return {"status": "success", "set_joints": set_joints,
                    "message": f"Set {len(set_joints)} joints"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def transform(self, prim_path, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Transform a USD model by applying position and scale.
        
        Args:
            prim_path (str): Path to the USD prim to transform
            position (tuple, optional): The position to set (x, y, z)
            scale (tuple, optional): The scale to set (x, y, z)
            
        Returns:
            dict: Result information
        """
        try:
            # Get the USD context
            stage = omni.usd.get_context().get_stage()
            
            # Get the prim
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                return {
                    "status": "error",
                    "message": f"Prim not found at path: {prim_path}"
                }
            
            # Initialize USDLoader
            loader = USDLoader()
            
            # Transform the model
            xformable = loader.transform(prim=prim, position=position, scale=scale)
            
            return {
                "status": "success",
                "message": f"Model at {prim_path} transformed successfully",
                "position": position,
                "scale": scale
            }
        except Exception as e:
            print(f"Error transforming model: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }
