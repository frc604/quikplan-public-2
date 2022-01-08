from enum import Enum

G = 9.81  # m/s/s


class StateVars(Enum):
    xIdx = 0
    yIdx = 1
    thetaIdx = 2
    velIdx = 3


class ControlVars(Enum):
    velDotIdx = 0
    thetaDotIdx = 1
