import casadi as ca
import json
import numpy as np
import pylab as plt

from helpers import (
    anim_traj,
    in2m,
    rotate_around_origin,
    plot_traj,
    create_obstacles,
    interp_state_vector,
    write_to_csv,
)
from robot import Robot
from constants import OptiParams

from joint_galactic_search_helper import (
    create_path,
    apply_wheel_force_friction_constraints,
)

OBSTACLES = []

def plan(robot, plot=False):
    # Compute each trajectory separately
    total_time = {
        'red': 0,
        'blue': 0,
    }
    solution = {}
    for path_id in ['A', 'B']:
        for color in ['red', 'blue']:
            conservative_robot = Robot(OptiParams(0.5, 1.0, 3000, 0.1))  # Conservative robot to find initial solution easier
            targets = create_obstacles(f"galactic-search-{path_id}-{color}")
            opti = ca.Opti()
            path = create_path(
                opti,
                conservative_robot,
                targets,
                constrain_start=(in2m(30), in2m(90)),
                constrain_end=(in2m(360), in2m(90), np.pi),
            )
            opti.minimize((path["T"]))

            # plot_traj(
            #     f"Initial Trajectory (Path {path_id} - {color})",
            #     path["x_init"],
            #     path["y_init"],
            #     path["theta_init"],
            #     OBSTACLES,
            #     conservative_robot.GEOMETRY,
            #     conservative_robot.AXIS_SIZE,
            #     targets=targets,
            # )

            # Solve non-linear program
            opti.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
            sol = opti.solve()

            # plot_traj(
            #     f"Path {path_id} - {color}",
            #     sol.value(path["xpos"]),
            #     sol.value(path["ypos"]),
            #     sol.value(path["theta"]),
            #     OBSTACLES,
            #     conservative_robot.GEOMETRY,
            #     conservative_robot.AXIS_SIZE,
            #     targets=targets,
            # )

            # Solve the problem again, but this time with wheel force & friction limit constraints
            opti2 = ca.Opti()
            path2 = create_path(opti2, robot, targets)
            opti2.minimize((path2["T"]))
            apply_wheel_force_friction_constraints(opti2, sol, robot, path2, prev_path=path)

            # Solve again
            opti2.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
            sol2 = opti2.solve()
            total_time[color] += sol2.value(path2['T'])

            # plot_traj(
            #     f"Path {path_id} - {color}",
            #     sol2.value(path2["xpos"]),
            #     sol2.value(path2["ypos"]),
            #     sol2.value(path2["theta"]),
            #     OBSTACLES,
            #     robot.GEOMETRY,
            #     robot.AXIS_SIZE,
            #     targets=targets,
            # )

            # Save path
            solution[path_id + color] = (sol2, path2)

    print(total_time)

    # Choose the faster path
    if total_time['red'] < total_time['blue']:
        # Recompute blue to start at red
        color_to_keep = 'red'
        color = 'blue'
    else:
        # Recompute red to start at blue
        color_to_keep = 'blue'
        color = 'red'

    # Reset and recompute
    total_time[color] = 0
    for path_id in ['A', 'B']:
        sol0, path0 = solution[path_id + color_to_keep]
        start_constraint = (sol0.value(path0['xpos'])[0], sol0.value(path0['ypos'])[0])

        conservative_robot = Robot(OptiParams(0.5, 1.0, 3000, 0.1))  # Conservative robot to find initial solution easier
        targets = create_obstacles(f"galactic-search-{path_id}-{color}")
        opti = ca.Opti()
        path = create_path(
            opti,
            conservative_robot,
            targets,
            constrain_start=start_constraint,
            constrain_end=(in2m(360), in2m(90), np.pi),
        )
        opti.minimize((path["T"]))

        # plot_traj(
        #     f"Initial Trajectory (Path {path_id} - {color})",
        #     path["x_init"],
        #     path["y_init"],
        #     path["theta_init"],
        #     OBSTACLES,
        #     conservative_robot.GEOMETRY,
        #     conservative_robot.AXIS_SIZE,
        #     targets=targets,
        # )

        # Solve non-linear program
        opti.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
        sol = opti.solve()

        # plot_traj(
        #     f"Path {path_id} - {color}",
        #     sol.value(path["xpos"]),
        #     sol.value(path["ypos"]),
        #     sol.value(path["theta"]),
        #     OBSTACLES,
        #     conservative_robot.GEOMETRY,
        #     conservative_robot.AXIS_SIZE,
        #     targets=targets,
        # )

        # Solve the problem again, but this time with wheel force & friction limit constraints
        opti2 = ca.Opti()
        path2 = create_path(opti2, robot, targets, constrain_start=start_constraint)
        opti2.minimize((path2["T"]))
        apply_wheel_force_friction_constraints(opti2, sol, robot, path2, prev_path=path)

        # Solve again
        opti2.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
        sol2 = opti2.solve()
        total_time[color] += sol2.value(path2['T'])

        # plot_traj(
        #     f"Path {path_id} - {color}",
        #     sol2.value(path2["xpos"]),
        #     sol2.value(path2["ypos"]),
        #     sol2.value(path2["theta"]),
        #     OBSTACLES,
        #     robot.GEOMETRY,
        #     robot.AXIS_SIZE,
        #     targets=targets,
        # )

        # Save path
        solution[path_id + color] = (sol2, path2)

    for id, (sol, path) in solution.items():
        if plot:
            plot_traj(
                f"Path {id}",
                sol.value(path["xpos"]),
                sol.value(path["ypos"]),
                sol.value(path["theta"]),
                OBSTACLES,
                robot.GEOMETRY,
                robot.AXIS_SIZE,
                targets=create_obstacles(f"galactic-search-{id[0]}-{id[1:]}"),
            )

        N0 = path['N0']
        N1 = path['N1']
        N2 = path['N2']
        N3 = path['N3']
        T0 = path['T0']
        T1 = path['T1']
        T2 = path['T2']
        T3 = path['T3']
        dt0 = path['dt0']
        dt1 = path['dt1']
        dt2 = path['dt2']
        dt3 = path['dt3']
        times = np.concatenate(
            (
                np.linspace(0, sol.value(T0) - sol.value(dt0), N0),
                np.linspace(sol.value(T0), sol.value(T0) + sol.value(T1) - sol.value(dt1), N1),
                np.linspace(
                    sol.value(T0) + sol.value(T1),
                    sol.value(T0) + sol.value(T1) + sol.value(T2) - sol.value(dt2),
                    N2,
                ),
                np.linspace(
                    sol.value(T0) + sol.value(T1) + sol.value(T2),
                    sol.value(T0)
                    + sol.value(T1)
                    + sol.value(T2)
                    + sol.value(T3)
                    - sol.value(dt3),
                    N3,
                ),
            )
        )

        red_is_faster = total_time['red'] < total_time['blue']
        is_red = 'red' in id
        is_blue = 'blue' in id
        optimal = (red_is_faster and is_red) or (not red_is_faster and is_blue)
        write_to_csv(
            "trajectories/galactic_search2/search-mu_{:.1f}-torque_{:.1f}-rpm_{:.0f}-id_{}{}-time_{:.3f}".format(
                robot.MU,
                robot.MOTOR_MAX_TORQUE,
                robot.MOTOR_MAX_RPM,
                id,
                'P' if optimal else '',
                sol.value(path['T']),
            ),
            robot,
            times,
            sol.value(path['xpos']),
            sol.value(path['ypos']),
            sol.value(path['theta']),
            sol.value(path['vl']),
            sol.value(path['vr']),
            sol.value(path['al']),
            sol.value(path['ar']),
            sol.value(path['jl']),
            sol.value(path['jr']),
        )

    print(total_time)
    if plot:
        plt.show()


if __name__ == "__main__":
    robot = Robot()
    plan(robot, plot=True)
