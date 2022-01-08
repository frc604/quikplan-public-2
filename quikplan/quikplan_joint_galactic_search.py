import casadi as ca
import json
import numpy as np
import pylab as plt

import constants
from robot import Robot
from helpers import (
    anim_traj,
    in2m,
    rotate_around_origin,
    plot_traj,
    create_obstacles,
    interp_state_vector,
    write_to_csv,
)

from joint_galactic_search_helper import (
    create_path,
    apply_wheel_force_friction_constraints,
)


robot = Robot()
OBSTACLES = []

# Setup Optimization
opti = ca.Opti()

A_red_targets = create_obstacles("galactic-search-A-red")
A_blue_targets = create_obstacles("galactic-search-A-blue")
A_red_path = create_path(opti, robot, A_red_targets)
A_blue_path = create_path(opti, robot, A_blue_targets)

B_red_targets = create_obstacles("galactic-search-B-red")
B_blue_targets = create_obstacles("galactic-search-B-blue")
B_red_path = create_path(opti, robot, B_red_targets)
B_blue_path = create_path(opti, robot, B_blue_targets)

# Constrain start position
opti.subject_to(A_red_path["xpos"][0] == A_blue_path["xpos"][0])
opti.subject_to(A_red_path["ypos"][0] == A_blue_path["ypos"][0])
opti.subject_to(A_red_path["theta"][0] == A_blue_path["theta"][0])

opti.subject_to(B_red_path["xpos"][0] == B_blue_path["xpos"][0])
opti.subject_to(B_red_path["ypos"][0] == B_blue_path["ypos"][0])
opti.subject_to(B_red_path["theta"][0] == B_blue_path["theta"][0])

# Minimize time
# The best solution makes red and blue paths the same length.
opti.subject_to(
    A_red_path["T"] + B_red_path["T"] == A_blue_path["T"] + B_blue_path["T"]
)
opti.minimize((A_red_path["T"] + B_red_path["T"]))

# Plot initialization
plot_traj(
    "Initial Trajectory (Path A - Red)",
    A_red_path["x_init"],
    A_red_path["y_init"],
    A_red_path["theta_init"],
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=A_red_targets,
)
plot_traj(
    "Initial Trajectory (Path A - Blue)",
    A_blue_path["x_init"],
    A_blue_path["y_init"],
    A_blue_path["theta_init"],
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=A_blue_targets,
)
plot_traj(
    "Initial Trajectory (Path B - Red)",
    B_red_path["x_init"],
    B_red_path["y_init"],
    B_red_path["theta_init"],
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=B_red_targets,
)
plot_traj(
    "Initial Trajectory (Path B - Blue)",
    B_blue_path["x_init"],
    B_blue_path["y_init"],
    B_blue_path["theta_init"],
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=B_blue_targets,
)

# Solve non-linear program
opti.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
sol = opti.solve()

# Plot result without wheel force limits
plot_traj(
    "Before Wheel Force Limits (Path A - Red)",
    sol.value(A_red_path["xpos"]),
    sol.value(A_red_path["ypos"]),
    sol.value(A_red_path["theta"]),
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=A_red_targets,
)
plot_traj(
    "Before Wheel Force Limits (Path A - Blue)",
    sol.value(A_blue_path["xpos"]),
    sol.value(A_blue_path["ypos"]),
    sol.value(A_blue_path["theta"]),
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=A_blue_targets,
)
plot_traj(
    "Before Wheel Force Limits (Path B - Red)",
    sol.value(B_red_path["xpos"]),
    sol.value(B_red_path["ypos"]),
    sol.value(B_red_path["theta"]),
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=B_red_targets,
)
plot_traj(
    "Before Wheel Force Limits (Path B - Blue)",
    sol.value(B_blue_path["xpos"]),
    sol.value(B_blue_path["ypos"]),
    sol.value(B_blue_path["theta"]),
    OBSTACLES,
    robot.GEOMETRY,
    robot.AXIS_SIZE,
    targets=B_blue_targets,
)

# Solve the problem again, but this time with wheel force & friction limit constraints
apply_wheel_force_friction_constraints(opti, sol, robot, A_red_path)
apply_wheel_force_friction_constraints(opti, sol, robot, A_blue_path)
apply_wheel_force_friction_constraints(opti, sol, robot, B_red_path)
apply_wheel_force_friction_constraints(opti, sol, robot, B_blue_path)

# Solve again
sol = opti.solve()

# Plot final result
for name, path, targets in [
    ("Path A - Red", A_red_path, A_red_targets),
    ("Path A - Blue", A_blue_path, A_blue_targets),
    ("Path B - Red", B_red_path, B_red_targets),
    ("Path B - Blue", B_blue_path, B_blue_targets),
]:
    X = path["X"]
    U = path["U"]
    N0 = path["N0"]
    N1 = path["N1"]
    N2 = path["N2"]
    N3 = path["N3"]
    dt0 = path["dt0"]
    dt1 = path["dt1"]
    dt2 = path["dt2"]
    dt3 = path["dt3"]
    T0 = path["T0"]
    T1 = path["T1"]
    T2 = path["T2"]
    T3 = path["T3"]
    T = path["T"]
    xpos = path["xpos"]
    ypos = path["ypos"]
    theta = path["theta"]
    vl = X[constants.StateVars.vlIdx.value, :]
    vr = X[constants.StateVars.vrIdx.value, :]
    al = X[constants.StateVars.alIdx.value, :]
    ar = X[constants.StateVars.arIdx.value, :]
    jl = U[constants.ControlVars.jlIdx.value, :]
    jr = U[constants.ControlVars.jrIdx.value, :]

    plot_traj(
        f"Final Result ({name})",
        sol.value(xpos),
        sol.value(ypos),
        sol.value(theta),
        OBSTACLES,
        robot.GEOMETRY,
        robot.AXIS_SIZE,
        targets=targets,
    )

    plt.figure()
    times = np.concatenate(
        (
            np.linspace(0, sol.value(T0) - sol.value(dt0), N0),
            np.linspace(
                sol.value(T0), sol.value(T0) + sol.value(T1) - sol.value(dt1), N1
            ),
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

    write_to_csv(
        "trajectories/search/search({})-mu_{:.1f}-torque_{:.1f}-rpm_{:.0f}-time_{:.3f}".format(
            name, robot.MU, robot.MOTOR_MAX_TORQUE, robot.MOTOR_MAX_RPM, sol.value(T),
        ),
        robot,
        times,
        sol.value(xpos),
        sol.value(ypos),
        sol.value(theta),
        sol.value(vl),
        sol.value(vr),
        sol.value(al),
        sol.value(ar),
    )

    plt.plot(
        times, sol.value(vl)[:-1], label="Left Wheel Velocity", linewidth=4, color="red"
    )
    plt.plot(
        times,
        sol.value(vr)[:-1],
        label="Right Wheel Velocity",
        linewidth=4,
        color="blue",
    )
    plt.plot(
        times, sol.value(al)[:-1], label="Left Wheel Acceleration", color="firebrick"
    )
    plt.plot(
        times, sol.value(ar)[:-1], label="Right Wheel Acceleration", color="royalblue"
    )
    plt.plot(
        times,
        sol.value(jl),
        label="Left Wheel Jerk",
        linestyle="--",
        color="lightcoral",
    )
    plt.plot(
        times,
        sol.value(jr),
        label="Right Wheel Jerk",
        linestyle="--",
        color="cornflowerblue",
    )
    plt.legend(loc="lower left")
    plt.xlabel("Time (s)")
    plt.title(name)

    lon_fl, lon_fr = robot.get_longitudinal_wheel_forces(al, ar)
    lat_f = robot.get_lateral_wheel_force(vl, vr)
    plt.figure()
    plt.plot(times, sol.value(lon_fl)[:-1], label="Longitudinal Force (left)")
    plt.plot(times, sol.value(lon_fr)[:-1], label="Longitudinal Force (right)")
    plt.plot(times, sol.value(lat_f)[:-1], label="Centripetal Force")
    plt.legend(loc="lower left")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(name)

    plt.figure()
    plt.plot(
        times,
        np.sqrt(sol.value(lon_fl) ** 2 + sol.value(lat_f) ** 2)[:-1],
        label="Total Force (Left)",
    )
    plt.plot(
        times,
        np.sqrt(sol.value(lon_fr) ** 2 + sol.value(lat_f) ** 2)[:-1],
        label="Total Force (Right)",
    )
    plt.legend(loc="lower left")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(name)

    interp_x = interp_state_vector(times, sol.value(xpos), 0.02)
    interp_y = interp_state_vector(times, sol.value(ypos), 0.02)
    interp_theta = interp_state_vector(times, sol.value(theta), 0.02)

    plot_traj(
        f"Interp ({name})",
        interp_x,
        interp_y,
        interp_theta,
        OBSTACLES,
        robot.GEOMETRY,
        robot.AXIS_SIZE,
        targets=targets,
    )

    anim = anim_traj(
        f"Final Result ({name})",
        interp_x,
        interp_y,
        interp_theta,
        OBSTACLES,
        robot.GEOMETRY,
        robot.AXIS_SIZE,
        20,  # milliseconds
        targets=targets,
    )
    # anim.save(f"quikplan_{name}.gif", writer="pillow", fps=50)

    print(name)
    print(sol.value(T0))
    print(sol.value(T1))
    print(sol.value(T2))
    print(sol.value(T3))
    print(sol.value(T))
plt.show()
