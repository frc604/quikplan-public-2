import casadi as ca
import json
import numpy as np
from numpy.linalg.linalg import solve
import csv
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
import constants


def plan(robot, plot=False):
    N0 = 50  # Number of control intervals per segment
    N1 = 100
    N2 = 100
    N3 = 50
    N = N0 + N1 + N2 + N3

    OBSTACLES = create_obstacles("bounce-simplified")
    TARGETS = [
        (in2m(90), in2m(150), in2m(2.5)),  # x, y, r
        (in2m(180), in2m(160), in2m(2.5)),
        (in2m(270), in2m(170), in2m(2.5)),
    ]
    FINISH_LINE_BUFFER = 0.1

    # Setup Optimization
    opti = ca.Opti()

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

    # Minimize time
    opti.minimize(T)

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
    opti.subject_to(xpos[0] == in2m(60) - robot.LENGTH / 2)
    opti.subject_to(ypos[0] == in2m(90))
    opti.subject_to(theta[0] == 0)
    opti.subject_to(vl[0] == 0)
    opti.subject_to(vr[0] == 0)
    opti.subject_to(al[0] == 0)
    opti.subject_to(ar[0] == 0)
    opti.subject_to(jl[0] == 0)
    opti.subject_to(jr[0] == 0)
    # End
    robot.apply_finish_line_constraints(
        opti,
        xpos[-1],
        ypos[-1],
        theta[-1],
        (
            (in2m(300) + FINISH_LINE_BUFFER, in2m(60)),
            (in2m(300) + FINISH_LINE_BUFFER, in2m(120)),
        ),
        "right",
        backwards=True,
    )

    # Targets
    dx = xpos[N0 - 1] - TARGETS[0][0]
    dy = ypos[N0 - 1] - TARGETS[0][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= robot.LENGTH / 2)

    dx = xpos[(N0 + N1) - 1] - TARGETS[1][0]
    dy = ypos[(N0 + N1) - 1] - TARGETS[1][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= robot.LENGTH / 2)

    dx = xpos[(N0 + N1 + N2) - 1] - TARGETS[2][0]
    dy = ypos[(N0 + N1 + N2) - 1] - TARGETS[2][1]
    dist = np.sqrt((dx * dx + dy * dy))
    opti.subject_to(dist <= robot.LENGTH / 2)

    # Obstacles
    robot.apply_obstacle_constraints(opti, xpos, ypos, theta, OBSTACLES)

    # Time constraints
    opti.subject_to(T0 >= 0)
    opti.subject_to(T1 >= 0)
    opti.subject_to(T2 >= 0)
    opti.subject_to(T3 >= 0)

    # Compute initial guess for segment 0 from init traj
    with open("init_traj/bounce_00.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N0 + 1) / init_traj_len)

    x_init = []
    y_init = []
    theta_init = []
    x1 = in2m(30)
    y1 = in2m(90)
    theta1 = 0.0
    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1)
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint))
        y_init += list(np.linspace(y1, y2, n_per_waypoint))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = N0 - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 1 from init traj
    with open("init_traj/bounce_01.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N1 + 1) / init_traj_len)

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) - np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint))
        y_init += list(np.linspace(y1, y2, n_per_waypoint))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = (N0 + N1) - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 2 from init traj
    with open("init_traj/bounce_02.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N2 + 1) / init_traj_len)

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1)
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint))
        y_init += list(np.linspace(y1, y2, n_per_waypoint))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint))

        x1, y1, theta1 = x2, y2, theta2

    # Fill in remainder, if any
    remainder = (N0 + N1 + N2) - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Compute initial guess for segment 3 from init traj
    with open("init_traj/bounce_03.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N3 + 1) / init_traj_len)

    for i in range(init_traj_len):
        x2, y2 = init_traj[i]
        theta2 = np.arctan2(y2 - y1, x2 - x1) - np.pi  # We are driving backwards
        theta2 = np.unwrap([theta1, theta2])[-1]

        x_init += list(np.linspace(x1, x2, n_per_waypoint))
        y_init += list(np.linspace(y1, y2, n_per_waypoint))
        theta_init += list(np.linspace(theta1, theta2, n_per_waypoint))

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

    if plot:
        # Plot initialization
        plot_traj(
            "Initial Trajectory",
            x_init,
            y_init,
            theta_init,
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
            targets=TARGETS,
        )

    # Solve non-linear program
    opti.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
    sol = opti.solve()

    if plot:
        # Plot result without wheel force limits
        plot_traj(
            "Before Wheel Force Limits",
            sol.value(xpos),
            sol.value(ypos),
            sol.value(theta),
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
            targets=TARGETS,
        )

    # Solve the problem again, but this time with wheel force & friction limit constraints
    robot.apply_wheel_force_constraints(opti, al, ar)
    robot.apply_wheel_friction_constraints(opti, vl, vr, al, ar)

    # Copy over X, U, and T to initialize
    opti.set_initial(X, sol.value(X))
    opti.set_initial(U, sol.value(U))
    opti.set_initial(T0, sol.value(T0))
    opti.set_initial(T1, sol.value(T1))
    opti.set_initial(T2, sol.value(T2))
    opti.set_initial(T3, sol.value(T3))
    sol = opti.solve()

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

    if plot:
        # Plot final result
        plot_traj(
            "Final Result",
            sol.value(xpos),
            sol.value(ypos),
            sol.value(theta),
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
            targets=TARGETS,
        )

        plt.figure()
        plt.plot(
            times,
            sol.value(vl)[:-1],
            label="Left Wheel Velocity",
            linewidth=4,
            color="red",
        )
        plt.plot(
            times,
            sol.value(vr)[:-1],
            label="Right Wheel Velocity",
            linewidth=4,
            color="blue",
        )
        plt.plot(
            times,
            sol.value(al)[:-1],
            label="Left Wheel Acceleration",
            color="firebrick",
        )
        plt.plot(
            times,
            sol.value(ar)[:-1],
            label="Right Wheel Acceleration",
            color="royalblue",
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

        lon_fl, lon_fr = robot.get_longitudinal_wheel_forces(al, ar)
        lat_f = robot.get_lateral_wheel_force(vl, vr)
        plt.figure()
        plt.plot(times, sol.value(lon_fl)[:-1], label="Longitudinal Force (left)")
        plt.plot(times, sol.value(lon_fr)[:-1], label="Longitudinal Force (right)")
        plt.plot(times, sol.value(lat_f)[:-1], label="Centripetal Force")
        plt.legend(loc="lower left")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")

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

        interp_x = interp_state_vector(times, sol.value(xpos), 0.02)
        interp_y = interp_state_vector(times, sol.value(ypos), 0.02)
        interp_theta = interp_state_vector(times, sol.value(theta), 0.02)

        interp_v = interp_state_vector(
            times, ((sol.value(vl) + sol.value(vr)) / 2), 0.02
        )

        interp_vl = interp_state_vector(times, sol.value(vl), 0.02)
        interp_vr = interp_state_vector(times, sol.value(vr), 0.02)
        interp_v_2 = (interp_vl + interp_vr) / 2

        plt.figure()
        plt.plot(
            np.arange(0, times[-1], 0.02),
            interp_v,
            label="Calculate V Before",
            linewidth=0.1,
            color="red",
        )
        plt.plot(
            np.arange(0, times[-1], 0.02),
            interp_v_2,
            label="Calculate V After",
            linewidth=0.1,
            color="blue",
        )

        plot_traj(
            "Interp",
            interp_x,
            interp_y,
            interp_theta,
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
            targets=TARGETS,
        )

        anim = anim_traj(
            "Final Result",
            interp_x,
            interp_y,
            interp_theta,
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
            20,  # milliseconds
            targets=TARGETS,
        )
        # anim.save("quikplan.gif", writer="pillow", fps=50)

        # plt.figure()
        # plt.spy(sol.value(ca.jacobian(opti.g, opti.x)))
        # plt.title("Jacobian")
        # plt.figure()
        # plt.spy(sol.value(ca.hessian(opti.f + ca.dot(opti.lam_g, opti.g), opti.x)[0]))
        # plt.title("Hessian")

        plt.show()

    print(sol.value(T0))
    print(sol.value(T1))
    print(sol.value(T2))
    print(sol.value(T3))
    print(sol.value(T))

    write_to_csv(
        "trajectories/bounce/bounce-buffer_{:.2f}-mu_{:.1f}-torque_{:.1f}-rpm_{:.0f}-time_{:.3f}".format(
            robot.OBSTACLE_BUFFER,
            robot.MU,
            robot.MOTOR_MAX_TORQUE,
            robot.MOTOR_MAX_RPM,
            sol.value(T),
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
        sol.value(jl),
        sol.value(jr),
    )


if __name__ == "__main__":
    robot = Robot()
    plan(robot, plot=True)
