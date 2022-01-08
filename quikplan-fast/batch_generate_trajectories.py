import itertools
from multiprocessing import Pool
import numpy as np

from robot import OptiParams, Robot
from quikplan_slalom import plan as plan_slalom
from quikplan_bounce import plan as plan_bounce
from quikplan_barrel_racing import plan as plan_barrel_racing


paths = ["slalom", "bounce", "barrel_racing"]
mus = np.linspace(0.3, 0.5, 3)
max_motor_torques = np.linspace(1.0, 2.0, 5)
max_motor_rpms = np.linspace(4700, 5300, 4)
obstacle_buffers = np.linspace(0.05, 0.2, 4)

planners = {
    "slalom": plan_slalom,
    "bounce": plan_bounce,
    "barrel_racing": plan_barrel_racing,
}


def plan_with_args(args):
    path, mu, max_torque, max_rpm, obstacle_buffer = args
    print((path, mu, max_torque, max_rpm, obstacle_buffer))

    robot = Robot(OptiParams(mu, max_torque, max_rpm, obstacle_buffer))
    try:
        planners[path](robot)
    except:
        print(f"Failed to plan: {(path, mu, max_torque, max_rpm, obstacle_buffer)}")


pool = Pool()
pool.map(
    plan_with_args,
    itertools.product(paths, mus, max_motor_torques, max_motor_rpms, obstacle_buffers),
)
