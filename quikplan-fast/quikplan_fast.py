import casadi as ca
import numpy as np
import pylab as plt

import robot
from helpers import create_obstacles, in2m


OBSTACLES = create_obstacles("hyperdrive")
ROBOT_RADIUS = in2m(20)

DEBUG = True
X_init = None
U_init = None
T_init = None
for is_init in [True, False]:
    N = 20
    opti = ca.Opti()

    X = opti.variable(4, N + 1)
    xpos = X[0, :]
    ypos = X[1, :]
    theta = X[2, :]
    vel = X[3, :]

    U = opti.variable(2, N)
    vel_dot = U[0, :]
    theta_dot = U[1, :]

    # Time variable
    T = opti.variable()
    dt = T / N
    opti.subject_to(T >= 0)

    # Setup cost function
    cost = T

    # Constrain kinodynamics
    def dynamics_model(x, u):
        theta_ = x[2]
        vel_ = x[3]
        vel_dot_ = u[0]
        theta_dot_ = u[1]

        return ca.vertcat(
            vel_ * np.cos(theta_),  # x_dot
            vel_ * np.sin(theta_),  # y_dot
            theta_dot_,  # theta_dot
            vel_dot_,  # vel_dot
        )

    for k in range(N):
        x_next = X[:, k] + dynamics_model(X[:, k], U[:, k]) * dt
        opti.subject_to(X[:, k + 1] == x_next)

    opti.subject_to(opti.bounded(-4, vel, 4))
    opti.subject_to(opti.bounded(-4, vel_dot, 4))
    opti.subject_to(opti.bounded(-1, theta_dot, 1))

    # Boundary conditions
    # Start
    START = (in2m(60), in2m(90), np.pi / 2)
    opti.subject_to(xpos[0] == START[0])
    opti.subject_to(ypos[0] == START[1])
    opti.subject_to(theta[0] == START[2])
    opti.subject_to(vel[0] == 0)
    # End
    GOAL = (in2m(120), in2m(150), 0)
    opti.subject_to(xpos[-1] == GOAL[0])
    opti.subject_to(ypos[-1] == GOAL[1])
    opti.subject_to(theta[-1] == GOAL[2])
    opti.subject_to(vel[-1] == 0)

    # Add obstacle cost
    for obx, oby, obr in OBSTACLES:
        for i in range(N + 1):
            cost += (
                50.0
                * np.fmax(
                    0,
                    (obr + ROBOT_RADIUS) ** 2
                    - (xpos[i] - obx) ** 2
                    - (ypos[i] - oby) ** 2,
                )
                ** 2
            )

    # Initialize
    if is_init:
        opti.set_initial(T, 1)
        opti.set_initial(xpos, np.linspace(START[0], GOAL[0], N + 1))
        opti.set_initial(ypos, np.linspace(START[1], GOAL[1], N + 1))
        opti.set_initial(theta, np.linspace(START[2], GOAL[2], N + 1))
    else:
        opti.set_initial(X, X_init)
        opti.set_initial(U, U_init)
        opti.set_initial(T, T_init)

    # Solve
    opti.solver("ipopt", {}, {"mu_init": 1e-3, "print_level": None if DEBUG else 0})
    opti.minimize(cost)
    sol = opti.solve()
    X_init = sol.value(X)
    U_init = sol.value(U)
    T_init = sol.value(T)

    print(sol.value(T))

    if not is_init:
        # Plot
        times = np.linspace(0, sol.value(T), N + 1)
        plt.plot(times, sol.value(xpos))
        plt.plot(times, sol.value(vel))
        plt.plot(times[:-1], sol.value(vel_dot))

        fig, ax = plt.subplots()
        plt.scatter(sol.value(xpos), sol.value(ypos))
        for obx, oby, obr in OBSTACLES:
            ax.add_artist(plt.Circle((obx, oby), obr, color="r"))

        plt.xlim(0, in2m(360))
        plt.ylim(0, in2m(180))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
