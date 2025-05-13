import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import numpy as np

from gello.agents.agent import Agent
from gello.robots.dynamixel import DynamixelRobot


@dataclass
class DynamixelRobotConfig:
    joint_ids: List[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: List[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: List[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    baudrate: int = 57600

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
            baudrate=self.baudrate,
        )

@dataclass
class TmRobotConfig(DynamixelRobotConfig):
    joint_ids: List[int] = field(default_factory=lambda: [11, 12, 13, 14, 15, 16])
    joint_offsets: List[float] = field(default_factory=lambda: [
        4 * np.pi / 2,
        2 * np.pi / 2,
        2 * np.pi / 2,
        0 * np.pi / 2,
        0 * np.pi / 2,
        2 * np.pi / 2,
    ])
    joint_signs: List[int] = field(default_factory=lambda: [1, -1, 1, -1, 1, 1])
    gripper_config: Tuple[int, int, int] = field(default_factory=lambda: (17, 212, 170))
    baudrate: int = 1_000_000

@dataclass
class XArmRobotConfig(DynamixelRobotConfig):
    joint_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    joint_offsets: List[float] = field(default_factory=lambda: [
        2 * np.pi / 2,
        2 * np.pi / 2,
        2 * np.pi / 2,
        2 * np.pi / 2,
        -1 * np.pi / 2 + 2 * np.pi,
        1 * np.pi / 2,
        1 * np.pi / 2,
    ])
    joint_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    gripper_config: Tuple[int, int, int] = field(default_factory=lambda: (8, 279, 279 - 50))
    baudrate: int = 57600

@dataclass
class PandaRobotConfig(DynamixelRobotConfig):
    joint_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    joint_offsets: List[float] = field(default_factory=lambda: [
        3 * np.pi / 2,
        2 * np.pi / 2,
        1 * np.pi / 2,
        4 * np.pi / 2,
        -2 * np.pi / 2 + 2 * np.pi,
        3 * np.pi / 2,
        4 * np.pi / 2,
    ])
    joint_signs: List[int] = field(default_factory=lambda: [1, -1, 1, 1, 1, -1, 1])
    gripper_config: Tuple[int, int, int] = field(default_factory=lambda: (8, 195, 152))
    baudrate: int = 57600

@dataclass
class URRobotConfig(DynamixelRobotConfig):
    joint_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    joint_offsets: List[float] = field(default_factory=lambda: [
        np.pi + 0 * np.pi,
        2 * np.pi + np.pi / 2,
        2 * np.pi + np.pi / 2,
        2 * np.pi + np.pi / 2,
        1 * np.pi,
        3 * np.pi / 2,
    ])
    joint_signs: List[int] = field(default_factory=lambda: [1, 1, -1, 1, 1, 1])
    gripper_config: Tuple[int, int, int] = field(default_factory=lambda: (7, 286, 248))
    baudrate: int = 57600

@dataclass
class URLeftRobotConfig(DynamixelRobotConfig):
    joint_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    joint_offsets: List[float] = field(default_factory=lambda: [
        0,
        1 * np.pi / 2 + np.pi,
        np.pi / 2 + 0 * np.pi,
        0 * np.pi + np.pi / 2,
        np.pi - 2 * np.pi / 2,
        -1 * np.pi / 2 + 2 * np.pi,
    ])
    joint_signs: List[int] = field(default_factory=lambda: [1, 1, -1, 1, 1, 1])
    gripper_config: Tuple[int, int, int] = field(default_factory=lambda: (7, 20, -22))
    baudrate: int = 57600

class GelloAgent(Agent):
    def __init__(
        self,
        type: str = "tm",
        port: str = "/dev/ttyUSB0",
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            assert type in ["tm", "xarm", "panda", "ur", "ur_left"], type
            if type == 'tm':
                config = TmRobotConfig()
            elif type == 'xarm':
                config = XArmRobotConfig()
            elif type == 'panda':
                config = PandaRobotConfig()
            elif type == 'ur':
                config = URRobotConfig()
            elif type == 'ur_left':
                config = URLeftRobotConfig()
            else:
                raise ValueError(f"Unknown robot type: {type}")
            
            self._robot = config.make_robot(port=port, start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        # print(np.rad2deg(self._robot.get_joint_state()))
        return self._robot.get_joint_state()
        dyna_joints = self._robot.get_joint_state()
        # current_q = dyna_joints[:-1]  # last one dim is the gripper
        current_gripper = dyna_joints[-1]  # last one dim is the gripper

        print(current_gripper)
        if current_gripper < 0.2:
            self._robot.set_torque_mode(False)
            return obs["joint_positions"]
        else:
            self._robot.set_torque_mode(False)
            return dyna_joints
