import json
import numpy as np

import constants
from helpers import (
    anim_traj,
    in2m,
    rotate_around_origin,
    plot_traj,
    create_obstacles,
    interp_state_vector,
)

FINISH_LINE_BUFFER = 0.5


def create_path(opti, robot, targets, constrain_start=None, constrain_end=None):
    # Number of control intervals per segment
    N0 = 50
    N1 = 50
    N2 = 50
    N3 = 50
    N = N0 + N1 + N2 + N3

    # State variables
    X = opti.variable(len(constants.StateVars), N + 1)
    xpos = X[constants.StateVars.xIdx.value, :]  # X position
    ypos = X[constants.StateVars.yIdx.value, :]  # Y-position
    theta = X[constants.StateVars.thetaIdx.value, :]  # Theta
    vl = X[constants.StateVars.vlIdx.value, :]  # Left wheel velocity
    vr = X[constants.StateVars.vrIdx.value, :]  # Right wheel velocity
    al = X[constants.StateVars.alIdx.value, :]  # Left wheel acceleration
    ar = X[constants.StateVars.arIdx.value, :]  # Right wheel acceleration

    # Control variables
    U = opti.variable(len(constants.ControlVars), N)
    jl = U[constants.ControlVars.jlIdx.value, :]  # Left wheel jerk
    jr = U[constants.ControlVars.jrIdx.value, :]  # Right wheel jerk

    # Total time variable per segment
    T0 = opti.variable()
    T1 = opti.variable()
    T2 = opti.variable()
    T3 = opti.variable()
    dt0 = T0 / N0  # length of one control interval
    dt1 = T1 / N1
    dt2 = T2 / N2
    dt3 = T3 / N3
    T = T0 + T1 + T2 + T3

    # Apply dynamic constriants
    for k in range(N0):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt0
        opti.subject_to(X[:, k + 1] == x_next)

    for k in range(N0, N0 + N1):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt1
        opti.subject_to(X[:, k + 1] == x_next)

    for k in range(N0 + N1, N0 + N1 + N2):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt2
        opti.subject_to(X[:, k + 1] == x_next)

    for k in range(N0 + N1 + N2, N0 + N1 + N2 + N3):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt3
        opti.subject_to(X[:, k + 1] == x_next)

    # Wheel constraints
    robot.apply_wheel_constraints(opti, vl, vr, al, ar, jl, jr)

    # Boundary conditions
    # Start
    if constrain_start is not None:
        opti.subject_to(xpos[0] == constrain_start[0])
        opti.subject_to(ypos[0] == constrain_start[1])
    else:
        opti.subject_to(xpos[0] == in2m(30) - robot.LENGTH / 2)
        opti.subject_to(ypos[0] < in2m(180) + robot.WIDTH / 4)
        opti.subject_to(ypos[0] > in2m(0) - robot.WIDTH / 4)
    opti.subject_to(theta[0] == np.pi)
    opti.subject_to(vl[0] == 0)
    opti.subject_to(vr[0] == 0)
    opti.subject_to(al[0] == 0)
    opti.subject_to(ar[0] == 0)
    opti.subject_to(jl[0] == 0)
    opti.subject_to(jr[0] == 0)
    # End
    if constrain_end is not None:
        opti.subject_to(xpos[-1] == constrain_end[0])
        opti.subject_to(ypos[-1] == constrain_end[1])
        opti.subject_to(theta[-1] == constrain_end[2])
    else:
        robot.apply_finish_line_constraints(
            opti,
            xpos[-1],
            ypos[-1],
            theta[-1],
            (
                (in2m(330) + FINISH_LINE_BUFFER, in2m(0)),
                (in2m(330) + FINISH_LINE_BUFFER, in2m(180)),
            ),
            "right",
            backwards=True,
        )
        opti.subject_to(ypos[-1] < in2m(180) + robot.WIDTH / 4)
        opti.subject_to(ypos[-1] > in2m(0) - robot.WIDTH / 4)

    # Targets
    dx = xpos[N0 - 1] - targets[0][0]
    dy = ypos[N0 - 1] - targets[0][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= in2m(4))

    dx = xpos[(N0 + N1) - 1] - targets[1][0]
    dy = ypos[(N0 + N1) - 1] - targets[1][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= in2m(4))

    dx = xpos[(N0 + N1 + N2) - 1] - targets[2][0]
    dy = ypos[(N0 + N1 + N2) - 1] - targets[2][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= in2m(4))

    # Time constraints
    opti.subject_to(T0 >= 0)
    opti.subject_to(T1 >= 0)
    opti.subject_to(T2 >= 0)
    opti.subject_to(T3 >= 0)

    # Compute initial guess for segment 0 from init traj
    init_traj = [[targets[0][0], targets[0][1]]]
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N0 + 1) / init_traj_len) - 1

    x_init = []
    y_init = []
    theta_init = []
    x1 = in2m(0)
    y1 = in2m(90)
    theta1 = np.pi
    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) + np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint, endpoint=False))
        y_init += list(np.linspace(y1, y2, n_per_waypoint, endpoint=False))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint, endpoint=False))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = N0 - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 1 from init traj
    init_traj = [[targets[1][0], targets[1][1]]]
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N1 + 1) / init_traj_len) - 1

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) + np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint, endpoint=False))
        y_init += list(np.linspace(y1, y2, n_per_waypoint, endpoint=False))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint, endpoint=False))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = (N0 + N1) - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 2 from init traj
    init_traj = [[targets[2][0], targets[2][1]]]
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N2 + 1) / init_traj_len) - 1

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) + np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint, endpoint=False))
        y_init += list(np.linspace(y1, y2, n_per_waypoint, endpoint=False))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint, endpoint=False))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = (N0 + N1 + N2) - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 3 from init traj
    init_traj = [[in2m(360), in2m(90)]]
    init_traj_len = len(init_traj)

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) + np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint, endpoint=False))
        y_init += list(np.linspace(y1, y2, n_per_waypoint, endpoint=False))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint, endpoint=False))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = (N0 + N1 + N2 + N3) - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Initial guess
    opti.set_initial(xpos, x_init)
    opti.set_initial(ypos, y_init)
    opti.set_initial(theta, theta_init)
    opti.set_initial(vl, 0)
    opti.set_initial(vr, 0)
    opti.set_initial(al, 0)
    opti.set_initial(ar, 0)
    opti.set_initial(jl, 0)
    opti.set_initial(jr, 0)
    opti.set_initial(T0, 5)
    opti.set_initial(T1, 5)
    opti.set_initial(T2, 5)
    opti.set_initial(T3, 5)

    return {
        "X": X,
        "U": U,
        "N0": N0,
        "N1": N1,
        "N2": N2,
        "N3": N3,
        "dt0": dt0,
        "dt1": dt1,
        "dt2": dt2,
        "dt3": dt3,
        "T0": T0,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "T": T,
        "xpos": xpos,
        "ypos": ypos,
        "theta": theta,
        "x_init": x_init,
        "y_init": y_init,
        "theta_init": theta_init,
        "vl": vl,
        "vr": vr,
        "al": al,
        "ar": ar,
        "jl": jl,
        "jr": jr,
    }


def apply_wheel_force_friction_constraints(opti, sol, robot, path, prev_path=None):
    X = path["X"]
    U = path["U"]
    T0 = path["T0"]
    T1 = path["T1"]
    T2 = path["T2"]
    T3 = path["T3"]

    vl = X[constants.StateVars.vlIdx.value, :]
    vr = X[constants.StateVars.vrIdx.value, :]
    al = X[constants.StateVars.alIdx.value, :]
    ar = X[constants.StateVars.arIdx.value, :]

    robot.apply_wheel_force_constraints(opti, al, ar)
    robot.apply_wheel_friction_constraints(opti, vl, vr, al, ar)

    # Copy over X, U, and T to initialize
    if prev_path is not None:
        Xinit = prev_path["X"]
        Uinit = prev_path["U"]
        T0init = prev_path["T0"]
        T1init = prev_path["T1"]
        T2init = prev_path["T2"]
        T3init = prev_path["T3"]
    else:
        Xinit = X
        Uinit = U
        T0init = T0
        T1init = T1
        T2init = T2
        T3init = T3

    opti.set_initial(X, sol.value(Xinit))
    opti.set_initial(U, sol.value(Uinit))
    opti.set_initial(T0, sol.value(T0init))
    opti.set_initial(T1, sol.value(T1init))
    opti.set_initial(T2, sol.value(T2init))
    opti.set_initial(T3, sol.value(T3init))
