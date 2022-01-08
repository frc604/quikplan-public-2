import sys, time, random
from PyQt5 import QtWidgets

import casadi as ca
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplashScreen,
    QVBoxLayout,
)
import qtmodern.styles
import qtmodern.windows
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import time as timeLib
import matplotlib.cm as cm
import numpy as np

import robot  # Only  needed to fix import issues
from helpers import QToaster, create_obstacles, in2m, LineBuilder, PATH_OBSTACLES


DEBUG = False
NETWORKTABLES = True
if NETWORKTABLES:
    from networktables import NetworkTables

ROBOT_RADIUS = in2m(25)

START_POS = (in2m(60), in2m(90), np.pi / 2)  # TODO: change based on selected path


def plan_path(start_state, end_state, obstacles, init_XU=None):
    N = 10
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

    opti.subject_to(opti.bounded(0, vel, 2.5))
    opti.subject_to(opti.bounded(-2, vel_dot, 2))
    opti.subject_to(opti.bounded(-np.pi, theta_dot, np.pi))

    # Curvature (k = theta_dot / vel)
    # Centripetal acceleration (MAX_AC >= vel^2 * k)
    # MAX_AC >= vel * theta_dot
    MAX_AC = 3  # m/s/s
    # Square both sides to keep everything positive
    opti.subject_to(MAX_AC * MAX_AC >= vel[1:] * vel[1:] * theta_dot * theta_dot)

    # Boundary conditions
    # Start
    opti.subject_to(xpos[0] == start_state[0])
    opti.subject_to(ypos[0] == start_state[1])
    opti.subject_to(theta[0] == start_state[2])
    opti.subject_to(vel[0] == start_state[3])
    # End
    opti.subject_to(xpos[-1] == end_state[0])
    opti.subject_to(ypos[-1] == end_state[1])
    end_theta_guess = np.unwrap(
        [
            start_state[2],
            np.arctan2(end_state[1] - start_state[1], end_state[0] - start_state[0]),
        ]
    )[-1]

    # Add obstacle cost
    for obx, oby, obr in obstacles:
        for i in range(N + 1):
            cost += (
                500.0
                * dt
                * vel[i]
                * np.fmax(
                    0,
                    (obr + ROBOT_RADIUS) ** 2
                    - (xpos[i] - obx) ** 2
                    - (ypos[i] - oby) ** 2,
                )
                ** 2
            )

    # Initialize
    opti.set_initial(T, 1)
    if init_XU is not None:
        init_X, init_U = init_XU
        xdim = init_X.shape[1]
        udim = init_U.shape[1]
        opti.set_initial(X[:, :xdim], init_X)
        opti.set_initial(U[:, :udim], init_U)

        opti.set_initial(
            xpos[xdim:], np.linspace(init_X[0, -1], end_state[0], N - xdim + 1)
        )
        opti.set_initial(
            ypos[xdim:], np.linspace(init_X[1, -1], end_state[1], N - xdim + 1)
        )
        opti.set_initial(
            theta[xdim:], np.linspace(init_X[2, -1], end_theta_guess, N - xdim + 1)
        )
    else:
        opti.set_initial(xpos, np.linspace(start_state[0], end_state[0], N + 1))
        opti.set_initial(ypos, np.linspace(start_state[1], end_state[1], N + 1))
        opti.set_initial(theta, np.linspace(start_state[2], end_theta_guess, N + 1))

    # Solve
    opti.solver("ipopt", {}, {"mu_init": 1e-3, "print_level": None if DEBUG else 0})
    opti.minimize(cost)
    sol = opti.solve()

    times = np.linspace(0, sol.value(T), N + 1)
    return times, sol.value(X), sol.value(U)


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Field Canvas
        self.canvas = FigureCanvas(plt.figure())
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        plt.axis("equal")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.xlim(-1, in2m(360) + 1)
        plt.ylim(-1, in2m(180) + 1)

        self.canvas.axes.plot([START_POS[0]], [START_POS[1]], marker="o")  # Start pos

        # Setup on-click callback to planner
        self.start_state = (START_POS[0], START_POS[1], START_POS[2], 0)
        self.start_time = 0.0
        self.obstacles = []
        self.init_XU = None

        def click_callback(event):
            try:
                times, X, U = plan_path(
                    self.start_state,
                    (event.xdata, event.ydata),
                    self.obstacles,
                    self.init_XU,
                )

                interp_x, interp_times = self.interp_state_vector(times, X[0, :], 0.1)
                interp_y, _ = self.interp_state_vector(times, X[1, :], 0.1)
                interp_theta, _ = self.interp_state_vector(times, X[2, :], 0.1)
                interp_vel, _ = self.interp_state_vector(times, X[3, :], 0.1)
                interp_angularvel, _ = self.interp_state_vector(
                    times, np.append(U[1, :], U[1, -1]), 0.1
                )

                if NETWORKTABLES:
                    self.publishTrajectory(
                        interp_times + self.start_time,
                        interp_x,
                        interp_y,
                        interp_theta,
                        interp_vel,
                        interp_angularvel,
                    )

                self.plot_states_as_arrow(interp_x, interp_y, interp_theta, interp_vel)
                self.start_state = X[:, 5]
                self.start_time += times[5]
                self.init_XU = X[:, 5:], U[:, 5:]
            except RuntimeError as e:
                print(e)
                QToaster.showMessage(
                    self,
                    "Invalid Trajectory!",
                    corner=QtCore.Qt.TopRightCorner,
                    icon=QtWidgets.QStyle.SP_MessageBoxCritical,
                    timeout=1000,
                )

        self.canvas.mpl_connect("button_press_event", click_callback)

        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        # Obstacles List
        obstaclesList = QListWidget()
        for obstacle in PATH_OBSTACLES:
            obstaclesList.addItem(QListWidgetItem(obstacle))
        obstaclesList.itemClicked.connect(self.add_obstacles)

        # Clear Button
        clearButton = QPushButton()
        clearButton.setText("Clear")
        clearButton.clicked.connect(self.clearTrajectory)

        # Layout
        main_layout = QHBoxLayout()

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        settings_layout = QVBoxLayout()
        settings_layout.addWidget(obstaclesList)
        settings_layout.addWidget(clearButton)

        main_layout.addLayout(settings_layout)
        main_layout.addLayout(plot_layout)

        self.setLayout(main_layout)

        if NETWORKTABLES:
            nt = NetworkTables.getTable("quikplanteleop")
            self.trajectoryTable = nt.getSubTable("trajectory")

            for key in self.trajectoryTable.getKeys():
                self.trajectoryTable.delete(key)

        self.canvas.draw()

    def clearTrajectory(self):
        for key in self.trajectoryTable.getKeys():
            self.trajectoryTable.delete(key)

        self.start_state = (START_POS[0], START_POS[1], START_POS[2], 0)
        self.start_time = 0.0
        self.init_XU = None

        self.canvas.axes.clear()

        plt.axis("equal")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.xlim(-1, in2m(360) + 1)
        plt.ylim(-1, in2m(180) + 1)

        self.canvas.axes.plot([START_POS[0]], [START_POS[1]], marker="o")  # Start pos
        self.canvas.draw()

    def publishTrajectory(self, times, xs, ys, thetas, vs, ws):
        for key in self.trajectoryTable.getKeys():
            self.trajectoryTable.delete(key)

        for (time, x, y, theta, v, w) in zip(times, xs, ys, thetas, vs, ws):
            self.trajectoryTable.putNumberArray(
                str(float(time)), list(map(float, [x, y, theta, v, w]))
            )

    def add_obstacles(self, item):
        self.obstacles = create_obstacles(item.text())
        for obj in self.canvas.axes.findobj(match=plt.Circle):
            obj.remove()

        for obx, oby, obr in self.obstacles:
            self.canvas.axes.add_artist(plt.Circle((obx, oby), obr, color="r"))
        self.canvas.draw()

    def interp_state_vector(self, times, states, new_dt):
        interp_times = np.arange(0, times[-1], new_dt)
        return np.interp(interp_times, times, states), interp_times

    def plot_states_as_arrow(self, xs, ys, thetas, velocities):
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if (not hasattr(self, "colorIndex")) or self.colorIndex == 9:
            self.colorIndex = 0

        us = np.cos(thetas) * (
            (velocities / np.linalg.norm(velocities)) * 1
        )  # 0.005 = max arrow length
        vs = np.sin(thetas) * ((velocities / np.linalg.norm(velocities)) * 1)
        self.canvas.axes.quiver(
            xs,
            ys,
            us,
            vs,
            color=colors[self.colorIndex],
            scale_units="xy",
            scale=1,
            width=0.005,
        )
        self.colorIndex += 1
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qtmodern.styles.dark(app)

    splash = QSplashScreen(
        QPixmap("resources/quikplan.png").scaledToHeight(
            400, QtCore.Qt.TransformationMode.SmoothTransformation
        )
    )
    splash.setWindowFlags(
        QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
    )
    splash.setEnabled(False)
    splash.show()

    main = Window()

    def showWindow():
        splash.close()
        main.show()

    if NETWORKTABLES:
        NetworkTables.initialize(server="localhost")

        i = 0
        while not NetworkTables.isConnected():
            i += 1
            if i / 10 == 1:
                print("Connecting.")
                i = 0
            timeLib.sleep(0.1)
            app.processEvents()

        print("Connected to {}!".format(NetworkTables.getRemoteAddress()))

        showWindow()
    else:
        QtCore.QTimer.singleShot(1000, showWindow)

    sys.exit(app.exec_())
