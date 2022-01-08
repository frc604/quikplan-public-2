from enum import Enum
from dataclasses import dataclass
from functools import wraps
from robot import DEFAULT_MAX_RPM, DEFAULT_MAX_TORQUE, DEFAULT_MU, DEFAULT_OBSTACLE_BUFFER

G = 9.81  # m/s/s


class StateVars(Enum):
    xIdx = 0
    yIdx = 1
    thetaIdx = 2
    vlIdx = 3
    vrIdx = 4
    alIdx = 5
    arIdx = 6

class ControlVars(Enum):
    jlIdx = 0
    jrIdx = 1

@dataclass(eq=True, frozen=True)
class OptiParams:
    mu: float = DEFAULT_MU
    max_torque: float = DEFAULT_MAX_TORQUE
    max_rpm: float = DEFAULT_MAX_RPM
    obstacle_buffer: float = DEFAULT_OBSTACLE_BUFFER

@dataclass(eq=True, frozen=True)
class Trajectory:
    type: str
    name: str
    params: OptiParams
    time: float = 0.0