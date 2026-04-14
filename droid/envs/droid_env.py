"""DROID-style data collection environment for IsaacLab 2.3.0.

Scene: Franka + Robotiq 2F-85 + 3 cameras (2 external + 1 wrist) + table scene.
Action: IK-relative SE(3) delta + binary gripper.
Observation: joint_pos(7), gripper_pos(1), 3x RGB images.
Control: 15 Hz (dt=1/120, decimation=8).
"""
from pathlib import Path

import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg

from isaaclab.controllers import DifferentialIKControllerCfg
from .franka_robotiq import FRANKA_ROBOTIQ_CFG

ASSET_PATH = Path(__file__).parent / "../assets"


# ---------------------------------------------------------------------------
# Custom binary action: 0 = open, 1 = close (matches DROID convention)
# ---------------------------------------------------------------------------
class BinaryGripperAction(BinaryJointPositionAction):
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        binary_mask = actions > 0.5 if actions.dtype != torch.bool else actions
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class BinaryGripperActionCfg(BinaryJointPositionActionCfg):
    class_type = BinaryGripperAction


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class DroidSceneCfg(InteractiveSceneCfg):
    """DROID-style scene: robot + 3 cameras + lighting."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/sphere_light",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = FRANKA_ROBOTIQ_CFG

    # --- 3 Cameras matching DROID placement (2 external + 1 wrist) ---
    external_cam_1 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_1",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=28.0,
            horizontal_aperture=5.376, vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

    external_cam_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=28.0,
            horizontal_aperture=5.376, vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8, focus_distance=28.0,
            horizontal_aperture=5.376, vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def load_scene(self, scene_id: str = "1"):
        """Load a DROID scene (1=cube+bowl, 2=can+mug, 3=banana+bin)."""
        usd_path = ASSET_PATH / f"scene{scene_id}.usd"
        self.scene_env = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        )


# ---------------------------------------------------------------------------
# Observation functions
# ---------------------------------------------------------------------------
def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    indices = [i for i, n in enumerate(robot.data.joint_names) if n in joint_names]
    return robot.data.joint_pos[:, indices]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"]
    pos = robot.data.joint_pos[:, indices]
    return pos / (np.pi / 4)  # normalize to [0, 1]


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------
@configclass
class ActionCfg:
    """IK-relative action space (for keyboard teleop)."""
    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_link8",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        scale=1.0,
    )
    gripper = BinaryGripperActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class JointPosActionCfg:
    """Joint position action space (for auto pipeline / scripted control)."""
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    gripper = BinaryGripperActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos)
        # Camera images are NOT in obs pipeline — published via ROS2 OmniGraph instead.
        # This avoids camera buffer race conditions and improves FPS.

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DroidEnvCfg(ManagerBasedRLEnvCfg):
    """DROID data collection environment config."""

    scene = DroidSceneCfg(num_envs=1, env_spacing=7.0)
    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = type("RewardsCfg", (), {})()
    terminations = TerminationsCfg()
    commands = type("CommandsCfg", (), {})()
    events = EventCfg()
    curriculum = type("CurriculumCfg", (), {})()

    def __post_init__(self):
        # 15 Hz control (dt=1/120, decimation=8) — original DROID spec
        self.episode_length_s = 600
        self.decimation = 8
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.rerender_on_reset = True

    def set_scene(self, scene_id: str = "1"):
        self.scene.load_scene(scene_id)


@configclass
class DroidEnvJointPosCfg(DroidEnvCfg):
    """DROID env with JointPosition action space (for scripted/auto pipelines)."""
    actions = JointPosActionCfg()
