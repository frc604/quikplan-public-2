import casadi as ca
import json
import csv
import numpy as np
import pylab as plt

from helpers import (
    anim_traj,
    in2m,
    interp_state_vector,
    rotate_around_origin,
    plot_traj,
    create_obstacles,
    write_to_csv,
)
from robot import Robot
import constants


def plan(robot, plot=False):
    N = 200  # Number of control intervals

    OBSTACLES = create_obstacles("slalom-simplified")
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

    # Total time variable
    T = opti.variable()
    dt = T / N  # length of one control interval

    # Minimize time
    opti.minimize(T)

    # Apply dynamic constriants
    for k in range(N):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt
        opti.subject_to(X[:, k + 1] == x_next)

    # Wheel constraints
    robot.apply_wheel_constraints(opti, vl, vr, al, ar, jl, jr)

    # Boundary conditions
    # Start
    opti.subject_to(xpos[0] == in2m(60) - robot.LENGTH / 2)
    opti.subject_to(ypos[0] == in2m(30))
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
            (in2m(60) - FINISH_LINE_BUFFER, in2m(60)),
            (in2m(60) - FINISH_LINE_BUFFER, in2m(120)),
        ),
        "left",
    )

    # Obstacles
    robot.apply_obstacle_constraints(opti, xpos, ypos, theta, OBSTACLES)

    # Time constraints
    opti.subject_to(T >= 0)

    # Compute initial guess from init traj
    with open("init_traj/slalom.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N + 1) / init_traj_len)

    x_init = []
    y_init = []
    theta_init = []
    x1 = in2m(30)
    y1 = in2m(30)
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
    remainder = N - len(x_init) + 1
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
    opti.set_initial(T, 10)

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
        )

    # Solve non-linear program
    opti.solver("ipopt", {}, {"mu_init": 1e-3})  # set numerical backend
    # opti.solver("knitro")  # set numerical backend
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
        )

    # Solve the problem again, but this time with wheel force & friction limit constraints
    robot.apply_wheel_force_constraints(opti, al, ar)
    robot.apply_wheel_friction_constraints(opti, vl, vr, al, ar)

    # Copy over X, U, and T to initialize
    opti.set_initial(X, sol.value(X))
    opti.set_initial(U, sol.value(U))
    opti.set_initial(T, sol.value(T))
    sol = opti.solve()

    times = np.linspace(0, sol.value(T), N)

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

        plot_traj(
            "Interp",
            interp_x,
            interp_y,
            interp_theta,
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
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
        )
        # anim.save("quikplan.gif", writer="pillow", fps=50)

        # plt.figure()
        # plt.spy(sol.value(ca.jacobian(opti.g, opti.x)))
        # plt.title("Jacobian")
        # plt.figure()
        # plt.spy(sol.value(ca.hessian(opti.f + ca.dot(opti.lam_g, opti.g), opti.x)[0]))
        # plt.title("Hessian")

        plt.show()

    print(sol.value(T))

    write_to_csv(
        "trajectories/slalom/slalom-buffer_{:.2f}-mu_{:.1f}-torque_{:.1f}-rpm_{:.0f}-time_{:.3f}".format(
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
