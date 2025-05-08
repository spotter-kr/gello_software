from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
from abc import ABC

class PositionMode(Enum):
    C = "C"
    T = "T"
    J = "J"

class MotionType(Enum):
    PTP = "PTP"
    LINE = "Line"

class MotionFormat(Enum):
    C = "CPP"
    J = "JPP"

class CommandType(Enum):
    POS = 0
    MOTION = 1
    GRIPPER = 2

@dataclass
class Command(ABC):
    command_type: CommandType

@dataclass
class Position(Command):
    command_type = CommandType.POS
    enable: Optional[bool] = None
    mode: PositionMode = PositionMode.J
    acc: int = 1000
    gain: int = 3
    protection: int = 20
    target: Optional[List[float]] = None

    def toscript(self) -> str:
        if self.enable is None:
            if self.target is None:
                return ''
            if len(self.target) != 6:
                return ''
            return f'Position({self.target[0]:.2f},{self.target[1]:.2f},{self.target[2]:.2f},{self.target[3]:.2f},{self.target[4]:.2f},{self.target[5]:.2f})'
        elif self.enable:
            return f'Position(true,"{self.mode.value}",{self.acc},{self.gain},{self.protection})'
        else:
            return f'Position(false)'

@dataclass
class Motion(Command):
    command_type = CommandType.MOTION
    motion_type: MotionType = MotionType.PTP
    motion_format: MotionFormat = MotionFormat.J
    speed: int = 100
    acc: int = 200
    blend: int = 0
    accuracy: bool = True
    target: List[float] = field(default_factory=list)
    
    def toscript(self) -> str:
        return f'{self.motion_type.value}("{self.motion_format.value}",{self.target[0]:.2f},{self.target[1]:.2f},{self.target[2]:.2f},{self.target[3]:.2f},{self.target[4]:.2f},{self.target[5]:.2f},{self.speed},{self.acc},{self.blend},{self.accuracy})'

@dataclass
class Gripper(Command):
    command_type = CommandType.GRIPPER
    target: float = 0    