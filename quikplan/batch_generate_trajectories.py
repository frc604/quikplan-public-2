import itertools
from multiprocessing import Pool
import numpy as np

from robot import OptiParams, Robot
from quikplan_slalom import plan as plan_slalom
from quikplan_bounce import plan as plan_bounce
from quikplan_barrel_racing import plan as plan_barrel_racing
from quikplan_galactic_search_fast import plan as plan_galactic_search


paths = ["galactic_search"]
mus = np.linspace(0.8, 1.0, 3)
max_motor_torques = [1.0]#np.linspace(1.0, 1.6, 4)
max_motor_rpms = np.linspace(3000, 3250, 2)
obstacle_buffers = [0.2]#np.linspace(0.05, 0.2, 4)

planners = {
    "slalom": plan_slalom,
    "bounce": plan_bounce,
    "barrel_racing": plan_barrel_racing,
    "galactic_search": plan_galactic_search,
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
    itertools.product(paths, mus, max_motor_torques, max_motor_rpms, [0.2]),
    # itertools.product(paths, mus, max_motor_torques, max_motor_rpms, obstacle_buffers),
)
