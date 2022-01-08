import casadi as ca
import json
import csv
import numpy as np
import pylab as plt

import constants
from robot import Robot
from helpers import (
    anim_traj,
    in2m,
    interp_state_vector,
    rotate_around_origin,
    plot_traj,
    create_obstacles,
    write_to_csv,
)


def plan(robot, plot=False):
    N = 50  # Number of control intervals

    OBSTACLES = create_obstacles("barrel-racing")
    FINISH_LINE_BUFFER = 0.1

    # Setup Optimization
    opti = ca.Opti()

    # State variables
    X = opti.variable(len(constants.StateVars), N + 1)
    xpos = X[constants.StateVars.xIdx.value, :]  # X position
    ypos = X[constants.StateVars.yIdx.value, :]  # Y-position
    theta = X[constants.StateVars.thetaIdx.value, :]  # Theta
    vel = X[constants.StateVars.velIdx.value, :]  # Forward velocity

    # Control variables
    U = opti.variable(len(constants.ControlVars), N)
    vel_dot = U[constants.ControlVars.velDotIdx.value, :]  # Forward acceleration
    theta_dot = U[constants.ControlVars.thetaDotIdx.value, :]  # Turn rate

    # Total time variable
    T = opti.variable()
    dt = T / N  # length of one control interval

    # Setup cost function
    cost = T

    # Apply dynamic constriants
    for k in range(N):
        x_next = X[:, k] + robot.dynamics_model(X[:, k], U[:, k]) * dt
        opti.subject_to(X[:, k + 1] == x_next)

    opti.subject_to(opti.bounded(-4, vel, 4))
    opti.subject_to(opti.bounded(-10, vel_dot, 10))
    opti.subject_to(opti.bounded(-1, theta_dot, 1))

    # Boundary conditions
    # Start
    opti.subject_to(xpos[0] == in2m(60) - robot.LENGTH / 2)
    opti.subject_to(ypos[0] == in2m(90))
    opti.subject_to(theta[0] == 0)
    opti.subject_to(vel[0] == 0)
    opti.subject_to(vel_dot[0] == 0)
    opti.subject_to(theta_dot[0] == 0)
    # End
    robot.apply_finish_line_constraints(
        opti,
        xpos[-1],
        ypos[-1],
        theta[-1],
        (
            (in2m(360) - FINISH_LINE_BUFFER, in2m(0)),
            (in2m(360) - FINISH_LINE_BUFFER, in2m(120)),
        ),
        "right",
    )
    opti.subject_to(ypos[-1] < 0)

    # Obstacles
    for i in range(N + 1):
        for obx, oby, obr in OBSTACLES:
            dx = xpos[i] - obx
            dy = ypos[i] - oby

            dist_sq = dx * dx + dy * dy
            cost += 1.0 * np.fmax(0, 0.5 - dist_sq) ** 2

    # Time constraints
    opti.subject_to(T >= 0)

    # Compute initial guess from init traj
    with open("init_traj/barrel_racing.json") as f:
        init_traj = json.load(f)
    init_traj_len = len(init_traj)
    n_per_waypoint = int((N + 1) / init_traj_len)

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
    remainder = N - len(x_init) + 1
    x_init += [x2] * remainder
    y_init += [y2] * remainder
    theta_init += [theta2] * remainder

    # Initial guess
    # opti.set_initial(xpos, x_init)
    opti.set_initial(xpos, np.linspace(0, in2m(300), N + 1))
    # opti.set_initial(ypos, y_init)
    # opti.set_initial(theta, theta_init)
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
    def opti_callback(i):
        plot_traj(
            f"Iteration {i}",
            opti.debug.value(xpos),
            opti.debug.value(ypos),
            opti.debug.value(theta),
            OBSTACLES,
            robot.GEOMETRY,
            robot.AXIS_SIZE,
        )
        plt.show()
        plt.close()

    opti.solver("ipopt", {}, {"mu_init": 1e-6})  # set numerical backend
    # opti.callback(opti_callback)
    opti.minimize(cost)
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
            sol.value(vel)[:-1],
            label="Velocity",
            linewidth=4,
            color="red",
        )
        plt.plot(
            times,
            sol.value(theta_dot),
            label="Turn rate",
            linewidth=4,
            color="blue",
        )
        plt.plot(
            times,
            sol.value(vel_dot),
            label="Acceleration",
            color="firebrick",
        )
        plt.legend(loc="lower left")
        plt.xlabel("Time (s)")

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

    # write_to_csv(
    #     "trajectories/barrel_racing/barrel_racing-buffer_{:.2f}-mu_{:.1f}-torque_{:.1f}-rpm_{:.0f}-time_{:.3f}".format(
    #         robot.OBSTACLE_BUFFER,
    #         robot.MU,
    #         robot.MOTOR_MAX_TORQUE,
    #         robot.MOTOR_MAX_RPM,
    #         sol.value(T),
    #     ),
    #     robot,
    #     times,
    #     sol.value(xpos),
    #     sol.value(ypos),
    #     sol.value(theta),
    #     sol.value(vl),
    #     sol.value(vr),
    #     sol.value(al),
    #     sol.value(ar),
    # )


if __name__ == "__main__":
    robot = Robot()
    plan(robot, plot=True)
